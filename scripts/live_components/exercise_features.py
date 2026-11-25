# Generated from frozen offline source by prepare_board_runtime.py.
from __future__ import annotations
import numpy as np

def longest_bad(x):
    e=np.flatnonzero(np.diff(np.r_[False,~x,False])); return int(max(e[1::2]-e[::2],default=0))

def features(k):
    # Input [time, 17, xyz(confidence)] in pixels; no face features.
    xy=k[:,5:17,:2].astype(float); conf=k[:,5:17,2]
    valid=np.isfinite(xy).all(-1)&(conf>=.3)
    # COCO shoulders 5/6 and hips 11/12 -> indices 0/1 and 6/7.
    torso=np.linalg.norm(xy[:,[0,1]]-xy[:,[6,7]],axis=-1)
    tv=valid[:,[0,1]]&valid[:,[6,7]]&(torso>1)
    if tv.sum()<len(k)*.5 or np.mean(valid.sum(1)>=6)<.9 or longest_bad(valid.sum(1)>=6)>4: return None
    scale=np.median(torso[tv]); center=np.median(xy[:,[6,7]][valid[:,[6,7]]],axis=0)
    z=np.where(valid[...,None],(xy-center)/scale,np.nan)
    # Per-joint spatial statistics and temporal variation, plus missing masks.
    result=[]
    for j in range(12):
        v=z[:,j]; ok=valid[:,j]; pair=ok[1:]&ok[:-1]
        if ok.any():
            result.extend(np.r_[v[ok].mean(0),v[ok].std(0),np.percentile(v[ok],10,axis=0),np.percentile(v[ok],90,axis=0)])
        else: result.extend([0.]*8)
        if pair.any():
            d=np.diff(v,axis=0)[pair]*15
            result.extend(np.r_[np.abs(d).mean(0),d.std(0)])
        else: result.extend([0.]*4)
        result.append(ok.mean())
    return np.asarray(result,dtype=np.float32)
