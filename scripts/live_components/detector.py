# Generated from frozen offline source by prepare_board_runtime.py.
from __future__ import annotations
import numpy as np
from collections import deque
from . import exercise_features as baseline

class AIExerciseDetector:
    """Fixed-FPS replay adapter. Causal trailing window; same features as training.
    Other is a non-target label, NOT trained idle. Invalid windows retain the
    confirmed label, matching v5's update semantics. No extra motion gate.
    """
    lying_ratio = 1.0

    def __init__(self, artifact, fps, seconds, confirm_sec=1.5):
        self.pipeline = artifact['pipeline']
        self.fps = float(fps)
        self.window_sec = float(seconds)
        self.confirm_sec = float(confirm_sec)
        self.span = round(seconds * self.fps)
        self.ids = np.floor(np.arange(round(seconds*15))*self.fps/15).astype(int)
        if self.fps < 15 or self.ids[-1] >= self.span:
            raise ValueError('Unsupported fixed-FPS sample grid')
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.span)
        self.candidate = self.candidate_since = self.confirmed = self.last_raw = None
        self.last_info = {}
        self.last_t = None

    def update(self, kp, t):
        if self.last_t is not None:
            dt = t - self.last_t
            if not np.isclose(dt, 1/self.fps, rtol=.02, atol=.001):
                raise ValueError(f'Nonuniform replay timestamp: {dt}; fixed FPS adapter only')
        self.last_t = t
        self.buf.append(np.asarray(kp, dtype=np.float32).copy())
        raw = None
        self.last_info = {'frames': len(self.buf)}
        if len(self.buf) == self.span:
            k = np.stack(self.buf)[self.ids]
            x = baseline.features(k)
            if x is not None and np.isfinite(x).all():
                p = self.pipeline.predict_proba(x[None])[0]
                raw = str(self.pipeline.classes_[np.argmax(p)]).lower()
                self.last_info.update({str(c): float(v) for c,v in zip(self.pipeline.classes_,p)})
        self.last_raw = raw
        if raw is None:
            return self.confirmed
        if raw != self.candidate:
            self.candidate, self.candidate_since = raw, t
        elif t - self.candidate_since >= self.confirm_sec and raw != self.confirmed:
            self.confirmed = raw
        return self.confirmed

class ExitHysteresisDetector(AIExerciseDetector):
    def __init__(self, artifact, fps, exit_sec):
        super().__init__(artifact, fps, 2, confirm_sec=1.5)
        self.exit_sec = float(exit_sec)
        if self.exit_sec < self.confirm_sec:
            raise ValueError('exit_sec must be >= confirm_sec')

    def update(self, kp, t):
        old = self.confirmed
        # Keep the original causal features, raw labels, candidate updates,
        # and missing-window handling. Delay only target -> Other transitions.
        super().update(kp, t)
        if (old in ('squat', 'pushup') and self.confirmed == 'other'
                and self.last_raw == 'other' and t - self.candidate_since < self.exit_sec):
            self.confirmed = old
        return self.confirmed
