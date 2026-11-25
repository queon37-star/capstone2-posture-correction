# Generated from frozen offline source by prepare_board_runtime.py.
from __future__ import annotations
import numpy as np
from collections import deque
from auto_detect_hailo_bridge import AutoDetectPipeline
from .pushup_branch import Branch

class UnifiedExercisePipeline(AutoDetectPipeline):
    """Preserve the original squat return contract; expose push_events separately.
    Caller supplies the same selected COCO17 pose to both branches. A new branch
    starts at each confirmed session; never replay across a consumed session.
    Squat's existing lying replay cut is deliberately separate from push-up's.
    """
    def __init__(self, base, fps, **kwargs):
        super().__init__(base, fps, **kwargs)
        if not self.gating: raise ValueError('Unified pipeline requires gating')
        self.push_branch = Branch(fps)
        self.push_recent = deque(maxlen=max(1, round(6*fps)))
        self.push_cut = -1
        self.push_events = []
        self.input_selected = False
        self.last_input = -1

    def _observe(self, m, frame_id):
        super()._observe(m, frame_id)
        if self.missing_run >= self.gap_frames:
            self.push_cut = max(self.push_cut, frame_id)

    def _publish_push(self, f, k, selected, published):
        b = self.push_branch
        n = len(b.events)
        b.feed(f, k, selected, published)
        if len(b.events) != n:
            e = dict(b.events[-1])
            e.update(exercise='pushup', session_id=self.current_session_id,
                     session_count=e['rep'], count=len(self.push_events)+1)
            self.push_events.append(e)

    def _start_session(self, exercise, frame_id, t):
        out = super()._start_session(exercise, frame_id, t)
        if exercise == 'pushup':
            onset = int(round(self.detector.candidate_since*self.fps))
            lo = max(self.push_cut+1, frame_id-round(6*self.fps),
                     onset-round(self.detector.window_sec*self.fps)+1)
            pre = [r for r in self.push_recent if r[0] >= lo]
            if pre and [r[0] for r in pre] != list(range(pre[0][0],frame_id)):
                raise RuntimeError('Noncontiguous push-up replay')
            self.push_branch.start(pre[0][0] if pre else frame_id)
            for f,k,s in pre: self._publish_push(f,k,s,frame_id)
        return out

    def _end_session(self, frame_id, t, reason):
        old = self.active_exercise
        if old == 'pushup':
            self.push_branch.stop()
            if frame_id >= 0: self._cut(frame_id-1)
        if frame_id >= 0:
            self.push_cut = max(self.push_cut, frame_id if reason=='person lost' else frame_id-1)
        return super()._end_session(frame_id,t,reason)

    def push_selected(self, kp, selected, *, frame_id):
        if self.closed or frame_id != self.last_input+1:
            raise ValueError('Expected contiguous open fixed-FPS stream')
        self.last_input = frame_id
        k = np.asarray(kp,dtype=np.float32).copy()
        if k.shape != (17,3): raise ValueError(k.shape)
        if not selected: k[:,:2]=np.nan; k[:,2]=0
        # Existing squat code masks confidence at the original threshold;
        # push-up gets the original confidence values as in its validated replay.
        self.input_selected = bool(selected)
        out = super().push(k,frame_id=frame_id,t=frame_id/self.fps)
        if self.active_exercise == 'pushup':
            self._publish_push(frame_id,k,bool(selected),frame_id)
        self.push_recent.append((frame_id,k,bool(selected)))
        return out
