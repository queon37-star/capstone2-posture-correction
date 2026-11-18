"""Observe the SAME Hailo-selected COCO17 pose as SquatStreamPipeline.

No new model inference, person selector, or squat counter. Events are written
separately to avoid changes to existing REP JSONL / dashboard consumers.

Optional (default OFF):
  SQUAT_AUTO_GATING=1   Feed the squat counter only during confirmed squat
                        sessions (see _push_gated). 0 = snapshot behaviour.
  SQUAT_KP_DUMP=<path>  Append every frame's COCO17 keypoints exactly as they
                        were handed to the squat pipeline, for offline replay
                        (see scripts/replay_kp_dump.py).  Fixed-size binary
                        records + '<path>.json' metadata sidecar.
"""
import atexit
import json
import os
import time
from collections import deque
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


class _CounterView:
    """What dashboards/logs see as pipeline.counter in gating mode.

    count is cumulative over all squat sessions (the UI must not drop to 0 when
    a new session creates a new SquatCounter); state is the current session's
    counter state, or INACTIVE outside a squat session. Every other attribute
    (missing_count, stand_hip, ... used by the trace CSV) is the current one.
    """

    def __init__(self, owner):
        self._owner = owner

    @property
    def count(self):
        return self._owner.detected_before + self._owner.base.counter.count

    @property
    def state(self):
        owner = self._owner
        return owner.base.counter.state if owner.active_exercise == 'squat' else 'INACTIVE'

    def __getattr__(self, name):
        return getattr(self._owner.base.counter, name)


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


class AutoDetectPipeline:
    """Wraps one SquatStreamPipeline; adds exercise detection and, optionally,
    session-based gating of the squat counter.

    Gating (SQUAT_AUTO_GATING=1) rules:
      * squat counter receives frames only while a squat session is active;
      * each squat session gets a NEW SquatStreamPipeline (make_base()), so no
        calibration/state from push-ups, idle or a previous person carries over;
      * at session start, recent frames (<= SQUAT_AUTO_REPLAY_SEC) are replayed
        so a rep finished before the detector confirmed 'squat' is kept;
      * replay never reaches back past: a lying run (push-up/plank posture,
        >= SQUAT_AUTO_LYING_FRAMES), a pose gap (>= SQUAT_AUTO_GAP_FRAMES),
        a person-lost reset, or the end of the previous squat session;
      * the new pipeline sees frame ids relative to the first replayed frame,
        so RepBoundaryBuffer clamps pre-context at that frame exactly as it does
        at frame 0 of a stream (no frame duplication or synthetic padding);
        rep frame fields are mapped back to absolute ids.
    """

    def __init__(self, base, fps, *, events_path=None, kp_dump_path=None,
                 make_base=None, gating=None, detector=None, replay_sec=None,
                 lying_frames=None, gap_frames=None):
        self.base = base
        self.fps = float(fps)
        self.detector = detector if detector is not None else ExerciseDetector()
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

        gating_value = os.getenv('SQUAT_AUTO_GATING', '0') if gating is None else gating
        if str(gating_value).strip() not in ('0', '1', 'False', 'True'):
            raise ValueError('SQUAT_AUTO_GATING must be 0 or 1')
        self.gating = str(gating_value).strip() in ('1', 'True')
        self.make_base = make_base
        if self.gating and make_base is None:
            raise ValueError('gating needs make_base() to create a fresh SquatStreamPipeline')
        self.replay_sec = float(replay_sec if replay_sec is not None
                                else _env_float('SQUAT_AUTO_REPLAY_SEC', 6.0))
        self.lying_frames = int(lying_frames if lying_frames is not None
                                else _env_float('SQUAT_AUTO_LYING_FRAMES', 5))
        self.gap_frames = int(gap_frames if gap_frames is not None
                              else _env_float('SQUAT_AUTO_GAP_FRAMES',
                                              max(10, int(self.fps * .5))))
        self.lying_ratio = float(getattr(self.detector, 'lying_ratio', 1.0))
        self.recent = deque(maxlen=max(1, int(round(self.replay_sec * self.fps))))
        self.cut_after = -1          # replay may only use frames with id > cut_after
        self.lying_run = 0
        self.missing_run = 0
        self.offset = 0              # absolute frame id of the current base's frame 0
        self.session_id = 0
        self.current_session_id = None
        self.detected_before = 0     # reps detected by finished squat sessions
        self.total_count = 0         # rep events published in this run
        self.session_count = 0       # rep events published in the current squat session
        self.session_last_rep = None
        self.last_replay = None
        self.max_replay_ms = 0.0
        print(f'AUTO enabled: timebase={self.timebase}, output={self.path}, '
              f'kp_dump={self.kp_dump.path if self.kp_dump else "off"}, '
              f'gating={int(self.gating)}'
              + (f' replay_sec={self.replay_sec:g} lying_frames={self.lying_frames} '
                 f'gap_frames={self.gap_frames}' if self.gating else ''), flush=True)

    # Dashboards read pipeline.counter / pipeline.event_count. Without gating
    # these are exactly the wrapped pipeline's objects, as before.
    @property
    def counter(self):
        return _CounterView(self) if self.gating else self.base.counter

    @property
    def event_count(self):
        return self.total_count if self.gating else self.base.event_count

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
        if self.gating:
            return self._push_gated(kp, frame_id, t)
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

    def _account(self, start, excluded=0.0):
        elapsed = time.perf_counter() - start - excluded
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

    # ------------------------------------------------------------------ gating
    def _push_gated(self, kp, frame_id, t):
        start = time.perf_counter()
        t = self._time(frame_id, t)
        if self.kp_dump is not None:
            self.kp_dump.write(frame_id, t, kp)
        kp = np.asarray(kp, dtype=np.float32)
        m = measure(kp)
        self._observe(m, frame_id)
        events = []
        excluded = 0.0

        if m is not None:
            self.seen_at = t
        if self.seen_at is not None and t - self.seen_at >= self.lost_sec:
            if self.active_exercise is not None:
                events += self._end_session(frame_id, t, 'person lost')
            self.detector.reset()
            self.last_confirmed = None
            self.seen_at = None
            self._cut(frame_id)

        label = self.detector.update(kp, t)
        if label != self.last_confirmed:
            self._emit('label_change', label, frame_id, t, raw=self.detector.last_raw)
            if self.active_exercise is not None:
                events += self._end_session(frame_id, t, 'label change')
            if label in ('squat', 'pushup'):
                replay_start = time.perf_counter()
                events += self._start_session(label, frame_id, t)
                excluded += time.perf_counter() - replay_start
            self.last_confirmed = label

        self.recent.append((frame_id, kp.copy()))
        if self.active_exercise == 'squat':
            pipeline_start = time.perf_counter()
            events += self._adopt(self.base.push(kp, frame_id=frame_id - self.offset))
            excluded += time.perf_counter() - pipeline_start
        self._account(start, excluded)
        if (frame_id + 1) % 150 == 0:
            print(f'AUTO frame={frame_id} raw={self.detector.last_raw} '
                  f'label={label} session={self.active_exercise}#{self.current_session_id} '
                  f'reps={self.session_count}/{self.total_count} '
                  f'info={self.detector.last_info} {self._perf_text()} '
                  f'replay_ms_max={self.max_replay_ms:.1f}', flush=True)
        return events

    def _cut(self, frame_id):
        self.cut_after = max(self.cut_after, int(frame_id))

    def _observe(self, m, frame_id):
        """Track lying (push-up/plank) runs and pose gaps for the replay cut."""
        if m is None:
            self.missing_run += 1          # a dropout inside a plank keeps the lying run
        else:
            self.missing_run = 0
            hands = m['hands_down']
            lying = (m['width_ratio'] >= self.lying_ratio
                     or (hands is not None and bool(hands)))
            self.lying_run = self.lying_run + 1 if lying else 0
        if self.lying_run >= self.lying_frames or self.missing_run >= self.gap_frames:
            self._cut(frame_id)

    def _replay_frames(self):
        frames = [item for item in self.recent if item[0] > self.cut_after]
        i = len(frames) - 1                # keep only the newest contiguous block
        while i > 0 and frames[i - 1][0] == frames[i][0] - 1:
            i -= 1
        return frames[max(i, 0):]

    def _start_session(self, exercise, frame_id, t):
        self.session_id += 1
        self.current_session_id = self.session_id
        self.active_exercise = exercise
        self._emit('session_start', exercise, frame_id, t, session_id=self.session_id)
        if exercise != 'squat':
            return []
        self.base = self.make_base()
        self.session_count = 0
        self.session_last_rep = None
        replay = self._replay_frames()
        self.offset = replay[0][0] if replay else int(frame_id)
        started = time.perf_counter()
        out = []
        for f, k in replay:
            out += self._adopt(self.base.push(k, frame_id=f - self.offset))
        ms = 1000 * (time.perf_counter() - started)
        self.max_replay_ms = max(self.max_replay_ms, ms)
        self.last_replay = {
            'session_id': self.session_id, 'frames': len(replay),
            'first_frame': replay[0][0] if replay else None,
            'last_frame': replay[-1][0] if replay else None,
            'cut_after': self.cut_after, 'reps': len(out), 'ms': round(ms, 2)}
        print('AUTO_REPLAY', json.dumps(self.last_replay), flush=True)
        return out

    def _end_session(self, frame_id, t, reason):
        exercise = self.active_exercise
        out = []
        extra = {}
        if exercise == 'squat':
            # Completed reps still waiting for post-context are published with
            # post_context_complete=false; an unfinished rep is discarded.
            out = self._adopt(self.base.flush())
            self.detected_before += self.base.counter.count
            extra['session_reps'] = self.session_count
            self.base = self.make_base()   # untouched placeholder until next session
            self._cut(frame_id - 1)        # never replay frames this session consumed
        self._emit('session_end', exercise, frame_id, t, reason=reason,
                   session_id=self.current_session_id, **extra)
        self.active_exercise = None
        self.current_session_id = None
        return out

    def _adopt(self, events):
        """Map session-relative rep events to absolute frame ids + session fields."""
        out = []
        pre = getattr(getattr(self.base, 'assembler', None), 'pre_frames', 0)
        for event in events:
            event = dict(event)
            for key in ('start_frame', 'end_frame', 'detected_start_frame', 'detected_end_frame'):
                if key in event:
                    event[key] += self.offset
            self.session_count += 1
            self.total_count += 1
            event['count'] = self.total_count          # cumulative, as before gating
            event['session_count'] = self.session_count
            event['session_id'] = self.current_session_id
            if 'start_frame' in event and 'detected_start_frame' in event:
                # RepBoundaryBuffer keeps up to `pre` frames before the detected
                # start, but clamps at the session's first frame and right after
                # the previous published rep (back-to-back reps). Report both.
                pre_frames = event['detected_start_frame'] - event['start_frame']
                event['pre_context_frames'] = pre_frames
                event['pre_context_complete'] = pre_frames >= pre
                event['pre_context_cut'] = (
                    None if pre_frames >= pre
                    else 'session_start' if event['start_frame'] == self.offset
                    else 'previous_rep')
            self.session_last_rep = event
            out.append(event)
        return out

    def flush(self):
        if self.gating:
            result = []
            if not self.closed:
                if self.active_exercise is not None:
                    result = self._end_session(-1, -1, 'stream end')
                self.out.close()
                if self.kp_dump is not None:
                    self.kp_dump.close()
                self.closed = True
            return result
        result = self.base.flush()
        if not self.closed:
            if self.active_exercise is not None:
                self._emit('session_end', self.active_exercise, -1, -1, reason='stream end')
            self.out.close()
            if self.kp_dump is not None:
                self.kp_dump.close()
            self.closed = True
        return result
