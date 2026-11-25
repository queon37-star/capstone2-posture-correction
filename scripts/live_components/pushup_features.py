# Generated from frozen offline source by prepare_board_runtime.py.
from __future__ import annotations
import numpy as np

SIGNALS=('elbow_angle','shoulder_ankle_y','hip_alignment','hip_offset','hip_offset_abs','knee_angle','shoulder_wrist_x_abs','shoulder_wrist_y','torso_tilt','elbow_asymmetry')

STATS=('mean','std','p10','median','p90','p90_p10','first','last')

FEATURE_NAMES=[f'{s}_{t}' for s in SIGNALS for t in STATS]

def angle(a,b,c):
    u=a-b; v=c-b; den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
    out=np.full(len(a),np.nan); ok=np.isfinite(den)&(den>1e-6)
    out[ok]=np.degrees(np.arccos(np.clip((u[ok]*v[ok]).sum(1)/den[ok],-1,1)))
    return out

def mean_sides(a,b):
    x=np.stack([a,b],axis=1); ok=np.isfinite(x); n=ok.sum(1)
    return np.divide(np.where(ok,x,0).sum(1),n,out=np.full(len(x),np.nan),where=n>0)

def stats(a):
    a=np.asarray(a,float); valid=a[np.isfinite(a)]
    if not len(valid): return [np.nan]*8
    p10,med,p90=np.percentile(valid,[10,50,90]); n=max(1,int(np.ceil(.15*len(a))))
    def m(x):
        x=x[np.isfinite(x)]; return float(x.mean()) if len(x) else np.nan
    return [valid.mean(),valid.std(),p10,med,p90,p90-p10,m(a[:n]),m(a[-n:])]

def extract_features(kp):
    """Stable inference API: [T,17,3] pixel x,y,confidence, invalid frames masked.
    Uses only frames in the completed repetition. No clip-wide statistics.
    """
    k=np.asarray(kp,float)
    if k.ndim!=3 or k.shape[1:]!=(17,3) or len(k)<8: raise ValueError(f'Invalid sequence {k.shape}')
    xy=k[:,:,:2].copy(); xy[~(np.isfinite(k).all(2)&(k[:,:,2]>=.3))]=np.nan
    lens=np.concatenate([np.linalg.norm(xy[:,s]-xy[:,h],axis=1) for s,h in [(5,11),(6,12)]])
    lens=lens[np.isfinite(lens)&(lens>10)]
    if not len(lens): raise ValueError('no usable torso scale')
    scale=float(np.median(lens)); side=[]
    for s,e,w,h,j,a in [(5,7,9,11,13,15),(6,8,10,12,14,16)]:
        S,E,W,H,J,A=[xy[:,i] for i in (s,e,w,h,j,a)]
        d=A-S; d=d*np.where(d[:,0]<0,-1.,1.)[:,None]; norm=np.linalg.norm(d,axis=1); r=H-S
        cross=d[:,0]*r[:,1]-d[:,1]*r[:,0]
        off=np.divide(cross,norm*scale,out=np.full(len(k),np.nan),where=np.isfinite(norm)&(norm>1e-6))
        torso=H-S; tilt=np.degrees(np.arctan2(np.abs(torso[:,1]),np.abs(torso[:,0])))
        tilt[np.linalg.norm(torso,axis=1)<=1e-6]=np.nan
        side.append([angle(S,E,W),(S[:,1]-A[:,1])/scale,angle(S,H,A),off,np.abs(off),angle(H,J,A),np.abs(S[:,0]-W[:,0])/scale,(S[:,1]-W[:,1])/scale,tilt])
    values=[mean_sides(a,b) for a,b in zip(*side)]+[np.abs(side[0][0]-side[1][0])]
    x=np.array([t for a in values for t in stats(a)],dtype=np.float64)
    if x.shape!=(80,) or np.isinf(x).any() or not np.isfinite(x).any(): raise ValueError('invalid feature vector')
    return x

def quality(k,selected,fps):
    valid=np.isfinite(k).all(2)&(k[:,:,2]>=.3)&selected[:,None]
    chain=valid[:,[5,7,9,11,13,15]].all(1)|valid[:,[6,8,10,12,14,16]].all(1)
    edges=np.flatnonzero(np.diff(np.r_[False,~chain,False])); gaps=edges[1::2]-edges[::2]
    q=dict(selected_pct=float(100*selected.mean()),chain_valid_pct=float(100*chain.mean()),longest_chain_gap_sec=float(max(gaps,default=0)/fps))
    reasons=[]
    if len(k)<8 or (len(k)-1)/fps<.45: reasons.append('too_short')
    if selected.mean()<.95: reasons.append('selection_below_95pct')
    if chain.mean()<.90: reasons.append('chain_below_90pct')
    if q['longest_chain_gap_sec']>.5: reasons.append('chain_gap_above_0.5s')
    return q,reasons
