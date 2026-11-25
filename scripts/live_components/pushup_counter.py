# Generated from frozen offline source by prepare_board_runtime.py.
from __future__ import annotations
import numpy as np
from collections import deque

class DepthCounter:
    """Normalized shoulder-minus-ankle height (image y down).
    No future samples; event end is recovery threshold crossing, not GT boundary.
    """
    def __init__(self, fps, depth=.18, recovery=.75):
        self.fps=float(fps); self.depth=depth; self.recovery=recovery
        if not np.isfinite(self.fps) or self.fps<=0 or depth<=0 or not 0<recovery<=1: raise ValueError('invalid configuration')
        self.buf=deque(maxlen=max(1,round(.2*self.fps)))
        self.last_valid=None; self.last_frame=-1; self.top=self.bottom=self.start=self.peak=None
        self.filtered=np.nan; self.events=[]
    def reset_motion(self):
        self.buf.clear(); self.top=self.bottom=self.start=self.peak=None
    def update(self, value, frame):
        if frame!=self.last_frame+1: raise ValueError('Expected contiguous frame IDs')
        self.last_frame=frame; self.filtered=np.nan
        if not np.isfinite(value):
            if self.last_valid is not None and (frame-self.last_valid)/self.fps>.35: self.reset_motion()
            return None
        self.last_valid=frame; self.buf.append(float(value)); v=float(np.median(self.buf)); self.filtered=v
        if self.top is None or (self.bottom is None and v<self.top): self.top=v; self.start=frame
        if self.bottom is None:
            if v-self.top>=self.depth and (frame-self.start)/self.fps>=.12: self.bottom=v; self.peak=frame
            return None
        if v>self.bottom: self.bottom=v; self.peak=frame
        if (frame-self.start)/self.fps>5:
            self.top=v; self.start=frame; self.bottom=self.peak=None; return None
        if self.bottom-v>=max(.12,self.recovery*(self.bottom-self.top)) and (frame-self.peak)/self.fps>=.12 and (frame-self.start)/self.fps>=.45:
            e=dict(rep=len(self.events)+1,start_frame=self.start,bottom_frame=self.peak,end_frame=frame,depth=self.bottom-self.top,duration_sec=(frame-self.start)/self.fps)
            self.events.append(e); self.top=v; self.start=frame; self.bottom=self.peak=None
            return e
        return None

class PushupCounter:
    """Single-frame COCO17 input, past-only torso-scale calibration."""
    def __init__(self, fps, depth=.18, recovery=.75):
        self.fps=float(fps); self.counter=DepthCounter(fps,depth,recovery)
        self.scales=deque(); self.last_scale=np.nan; self.value=np.nan
    def update(self, kp, frame, selected=True):
        k=np.asarray(kp,dtype=float)
        if k.shape!=(17,3): raise ValueError(f'Unexpected pose shape {k.shape}')
        ok=np.isfinite(k).all(1)&(k[:,2]>=.3)&bool(selected)
        while self.scales and (frame-self.scales[0][0])/self.fps>1.5: self.scales.popleft()
        for s,h in ((5,11),(6,12)):
            if ok[s] and ok[h]:
                d=float(np.linalg.norm(k[s,:2]-k[h,:2]))
                if d>10: self.scales.append((frame,d))
        self.last_scale=float(np.median([v for _,v in self.scales])) if self.scales else np.nan
        # Require >=3 past/current torso observations; no synthetic poses.
        vals=[k[s,1]-k[a,1] for s,a in ((5,15),(6,16)) if ok[s] and ok[a]]
        self.value=float(np.mean(vals)/self.last_scale) if len(self.scales)>=3 and vals and np.isfinite(self.last_scale) else np.nan
        return self.counter.update(self.value,frame)
