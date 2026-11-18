"""Observe the SAME Hailo-selected COCO17 pose as SquatStreamPipeline.

No new model inference, person selector, or squat counter. Events are written
separately to avoid changes to existing REP JSONL / dashboard consumers.
"""
import json
import os
import time
from pathlib import Path

from exercise_detector import ExerciseDetector, measure


class AutoDetectPipeline:
    def __init__(self, base, fps):
        self.base = base
        self.fps = float(fps)
        self.detector = ExerciseDetector()
        self.last_confirmed = None
        self.active_exercise = None
        self.seen_at = None
        self.t0 = time.perf_counter()
        self.timebase = os.getenv('SQUAT_AUTO_TIMEBASE', 'file').strip().lower()
        if self.timebase not in ('file', 'wall'):
            raise ValueError('SQUAT_AUTO_TIMEBASE must be file or wall')
        project_root = Path(__file__).resolve().parents[1]
        target = Path(os.getenv('SQUAT_AUTO_OUTPUT', 'results/hailo/auto_detect_events.jsonl')).expanduser()
        self.path = target if target.is_absolute() else project_root / target
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.out = self.path.open('w', encoding='utf-8')
        self.closed = False
        self.lost_sec = float(os.getenv('SQUAT_AUTO_LOST_SEC', '3.0'))
        print(f'AUTO enabled: timebase={self.timebase}, output={self.path}', flush=True)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _emit(self, type_, label, frame_id, t, **data):
        event = {'type': type_, 'exercise': label, 'frame_id': frame_id, 't': round(t, 3), **data}
        self.out.write(json.dumps(event, ensure_ascii=False) + '\n')
        self.out.flush()
        print('AUTO_EVENT', json.dumps(event, ensure_ascii=False), flush=True)

    def push(self, kp, *, frame_id):
        # Run the existing squat pipeline exactly once, with unchanged inputs.
        result = self.base.push(kp, frame_id=frame_id)
        t = (frame_id / self.fps if self.timebase == 'file'
             else time.perf_counter() - self.t0)

        # Cope with missing persons, as in the teammate's PC loop.
        if measure(kp) is not None:
            self.seen_at = t
        if self.seen_at is not None and t - self.seen_at >= self.lost_sec:
            if self.active_exercise is not None:
                self._emit('session_end', self.active_exercise, frame_id, t, reason='person lost')
                self.active_exercise = None
            self.detector.reset()
            self.last_confirmed = None
            self.seen_at = None

        label = self.detector.update(kp, t)
        if label != self.last_confirmed:
            self._emit('label_change', label, frame_id, t, raw=self.detector.last_raw)
            if self.active_exercise is not None:
                self._emit('session_end', self.active_exercise, frame_id, t, reason='label change')
            self.active_exercise = label if label in ('squat', 'pushup') else None
            if self.active_exercise is not None:
                self._emit('session_start', self.active_exercise, frame_id, t)
            self.last_confirmed = label
        if (frame_id + 1) % 150 == 0:
            print(f'AUTO frame={frame_id} raw={self.detector.last_raw} '
                  f'label={label} session={self.active_exercise} '
                  f'info={self.detector.last_info}', flush=True)
        return result

    def flush(self):
        result = self.base.flush()
        if not self.closed:
            if self.active_exercise is not None:
                self._emit('session_end', self.active_exercise, -1, -1, reason='stream end')
            self.out.close()
            self.closed = True
        return result
