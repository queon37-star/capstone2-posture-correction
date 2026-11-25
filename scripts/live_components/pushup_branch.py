# Generated from frozen offline source by prepare_board_runtime.py.
from __future__ import annotations
import numpy as np
from collections import deque
from .pushup_counter import PushupCounter
from .pushup_features import extract_features, quality, FEATURE_NAMES

THRESHOLDS = dict(hip_angle_p10_deg=160.0, hip_offset_abs_p90_torso=0.20,
                  elbow_angle_p10_deg=110.0, shoulder_range_torso=0.30)

def feedback(k, selected, fps):
    q, why = quality(k, selected, fps)
    h = o = np.nan
    if not why:
        try:
            x = extract_features(k)
            h = float(x[FEATURE_NAMES.index('hip_alignment_p10')])
            o = float(x[FEATURE_NAMES.index('hip_offset_abs_p90')])
        except ValueError as exc:
            why.append('feature:' + str(exc))
    if not np.isfinite([h, o]).all():
        if not why: why.append('alignment_unavailable')
        flag = -1
    else:
        flag = int(h < THRESHOLDS['hip_angle_p10_deg'] and o > THRESHOLDS['hip_offset_abs_p90_torso'])
    return dict(**q, quality_reason=';'.join(why), hip_alignment_p10=h,
                hip_offset_abs_p90=o, alignment_flag=flag,
                feedback={1:'alignment',0:'no_alignment_warning',-1:'unassessable'}[flag])

class Branch:
    """Independent session-local counter; all published frame IDs are absolute."""
    def __init__(self, fps):
        self.fps = fps
        self.events = []
        self.offset = None
        self.counter = None
        self.sessions = 0
        self.history = deque(maxlen=int(np.ceil(6*fps))+3)

    def start(self, frame):
        self.sessions += 1
        self.offset = int(frame)
        self.counter = PushupCounter(self.fps)
        self.history.clear()

    def stop(self):
        self.counter = None
        self.offset = None
        self.history.clear()

    def feed(self, frame, kp, selected, published_frame):
        if self.counter is None: raise RuntimeError('inactive branch')
        k = np.asarray(kp, dtype=np.float32).copy()
        if not selected:
            k[:, :2] = np.nan; k[:, 2] = 0
        self.history.append((frame, k, bool(selected)))
        event = self.counter.update(k, frame-self.offset, bool(selected))
        if event is None: return
        event = dict(event)
        for key in ('start_frame','bottom_frame','end_frame'):
            event[key] += self.offset
        st, ed = event['start_frame'], event['end_frame']
        hist = [r for r in self.history if st <= r[0] <= ed]
        if [r[0] for r in hist] != list(range(st,ed+1)):
            raise RuntimeError('counter interval missing history')
        q = feedback(np.stack([r[1] for r in hist]),np.array([r[2] for r in hist],bool),self.fps)
        self.events.append(dict(**event, **q, session_id=self.sessions,
                                total_count=len(self.events)+1, published_frame=published_frame,
                                replayed=ed < published_frame))
