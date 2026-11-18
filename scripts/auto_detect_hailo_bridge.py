"""Observe the SAME Hailo-selected COCO17 pose as SquatStreamPipeline.

No new model inference, person selector, or squat counter. Events are written
separately to avoid changes to existing REP JSONL / dashboard consumers.

Optional (default OFF):
  SQUAT_KP_DUMP=<path>  Append every frame's COCO17 keypoints exactly as they
                        were handed to the squat pipeline, for offline replay
                        (see scripts/replay_kp_dump.py).  Fixed-size binary
                        records + '<path>.json' metadata sidecar.
"""
import atexit
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from exercise_detector import ExerciseDetector, measure

KP_DUMP_VERSION = 1
# 8 + 8 + 1 + 7 + 17*3*4 = 228 bytes per frame (about 6.8 KB/s at 30 FPS).
KP_DUMP_DTYPE = np.dtype([
    ('frame_id', '<i8'),        # bridge frame id (0-based callback order)
    ('t', '<f8'),               # seconds; the exact value given to ExerciseDetector
    ('pose_valid', 'u1'),       # 1: a person's keypoints were passed, 0: missing_pose()
    ('reserved', 'u1', (7,)),
    ('kp', '<f4', (17, 3)),     # [x_px, y_px, confidence]; NaN x/y when missing
])


def read_kp_dump(path):
    """Return the structured array written by KeypointDump (works on partial files)."""
    path = Path(path)
    usable = path.stat().st_size // KP_DUMP_DTYPE.itemsize
    return np.fromfile(path, dtype=KP_DUMP_DTYPE, count=usable)


class KeypointDump:
    """Append-only keypoint recorder. Flushes every `flush_every` frames so an
    interrupted run (Ctrl+C, crash) loses at most about one second of data."""

    def __init__(self, path, *, fps, timebase, flush_every=30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.path.with_name(self.path.name + '.json')
        self.flush_every = max(1, int(flush_every))
        self.record = np.zeros(1, dtype=KP_DUMP_DTYPE)
        self.frames = 0
        self.invalid_frames = 0
        self.write_seconds = 0.0
        self.closed = False
        self.meta = {
            'format': 'capstone2-kp-dump', 'version': KP_DUMP_VERSION,
            'record_bytes': KP_DUMP_DTYPE.itemsize,
            'dtype': [list(map(str, d)) if len(d) == 2 else [d[0], d[1], list(d[2])]
                      for d in KP_DUMP_DTYPE.descr],
            'assumed_fps': fps, 'timebase': timebase,
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'env': {k: v for k, v in sorted(os.environ.items()) if k.startswith('SQUAT_')},
            'note': 'read with auto_detect_hailo_bridge.read_kp_dump(); '
                    'pose_valid=0 rows are missing_pose() frames.',
        }
        self._write_meta()
        self.file = self.path.open('wb')
        atexit.register(self.close)

    def _write_meta(self):
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=1),
                                  encoding='utf-8')

    def write(self, frame_id, t, kp):
        start = time.perf_counter()
        kp = np.asarray(kp, dtype=np.float32)
        rec = self.record
        rec['frame_id'] = frame_id
        rec['t'] = t
        valid = bool(np.isfinite(kp[:, :2]).any())
        rec['pose_valid'] = 1 if valid else 0
        rec['kp'][0] = kp
        self.file.write(rec.tobytes())
        self.frames += 1
        self.invalid_frames += 0 if valid else 1
        if self.frames % self.flush_every == 0:
            self.file.flush()
        self.write_seconds += time.perf_counter() - start

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.file.flush()
        self.file.close()
        self.meta.update({
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'frames': self.frames, 'pose_invalid_frames': self.invalid_frames,
            'mean_write_us': round(1e6 * self.write_seconds / max(self.frames, 1), 2),
        })
        self._write_meta()


class AutoDetectPipeline:
    def __init__(self, base, fps, *, events_path=None, kp_dump_path=None):
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

        def resolve(value):
            p = Path(value).expanduser()
            return p if p.is_absolute() else project_root / p

        self.path = resolve(events_path or os.getenv(
            'SQUAT_AUTO_OUTPUT', 'results/hailo/auto_detect_events.jsonl'))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.out = self.path.open('w', encoding='utf-8')
        self.closed = False
        self.lost_sec = float(os.getenv('SQUAT_AUTO_LOST_SEC', '3.0'))
        dump_value = kp_dump_path if kp_dump_path is not None else os.getenv('SQUAT_KP_DUMP', '')
        self.kp_dump = (KeypointDump(resolve(dump_value), fps=self.fps, timebase=self.timebase)
                        if str(dump_value).strip() else None)
        # Wrapper cost per frame, excluding the squat pipeline itself (reset every log line).
        self.perf_frames = 0
        self.perf_sum = 0.0
        self.perf_max = 0.0
        print(f'AUTO enabled: timebase={self.timebase}, output={self.path}, '
              f'kp_dump={self.kp_dump.path if self.kp_dump else "off"}', flush=True)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _emit(self, type_, label, frame_id, t, **data):
        event = {'type': type_, 'exercise': label, 'frame_id': frame_id, 't': round(t, 3), **data}
        self.out.write(json.dumps(event, ensure_ascii=False) + '\n')
        self.out.flush()
        print('AUTO_EVENT', json.dumps(event, ensure_ascii=False), flush=True)

    def _time(self, frame_id, t):
        if t is not None:            # offline replay supplies the recorded time
            return float(t)
        return (frame_id / self.fps if self.timebase == 'file'
                else time.perf_counter() - self.t0)

    def push(self, kp, *, frame_id, t=None):
        # Run the existing squat pipeline exactly once, with unchanged inputs.
        result = self.base.push(kp, frame_id=frame_id)
        wrapper_start = time.perf_counter()
        t = self._time(frame_id, t)
        if self.kp_dump is not None:
            self.kp_dump.write(frame_id, t, kp)

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
        self._account(wrapper_start)
        if (frame_id + 1) % 150 == 0:
            print(f'AUTO frame={frame_id} raw={self.detector.last_raw} '
                  f'label={label} session={self.active_exercise} '
                  f'info={self.detector.last_info} {self._perf_text()}', flush=True)
        return result

    def _account(self, start):
        elapsed = time.perf_counter() - start
        self.perf_frames += 1
        self.perf_sum += elapsed
        self.perf_max = max(self.perf_max, elapsed)

    def _perf_text(self):
        mean_us = 1e6 * self.perf_sum / max(self.perf_frames, 1)
        text = f'auto_us_mean={mean_us:.0f} auto_us_max={1e6 * self.perf_max:.0f}'
        if self.kp_dump is not None:
            text += (f' dump_us_mean={1e6 * self.kp_dump.write_seconds / max(self.kp_dump.frames, 1):.0f}'
                     f' dump_frames={self.kp_dump.frames}')
        self.perf_frames, self.perf_sum, self.perf_max = 0, 0.0, 0.0
        return text

    def flush(self):
        result = self.base.flush()
        if not self.closed:
            if self.active_exercise is not None:
                self._emit('session_end', self.active_exercise, -1, -1, reason='stream end')
            self.out.close()
            if self.kp_dump is not None:
                self.kp_dump.close()
            self.closed = True
        return result
