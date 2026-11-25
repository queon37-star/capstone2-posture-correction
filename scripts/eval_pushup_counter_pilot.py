#!/usr/bin/env python3
"""Causal push-up counter pilot, independent of exercise gating.
Run from pose_training_offline with this file in scripts/. Uses existing MV7 caches.
Manual counts are video-level references, NOT repetition boundaries or labels.
"""
from __future__ import annotations
import argparse, json, hashlib
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd

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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-dir',type=Path,default=Path('results/pushup/pilot'))
    p.add_argument('--work',type=Path,default=Path('data/preprocess/extra_datasets'))
    p.add_argument('--out-dir',type=Path,default=Path('results/pushup/counter_pilot'))
    p.add_argument('--depth',type=float,default=.18); p.add_argument('--recovery',type=float,default=.75)
    a=p.parse_args()
    import preprocess_rehab24 as pre
    from extract_video_kp_cache import KP_DUMP_DTYPE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from zipfile import ZipFile,ZIP_DEFLATED
    manual=pd.read_csv(a.pilot_dir/'manual_counts.csv'); inventory=pd.read_csv(a.pilot_dir/'pilot_inventory.csv')
    if len(manual)!=6 or manual.sample_id.duplicated().any() or set(manual.sample_id)!=set(inventory.sample_id): raise ValueError('Expected six exact pilot IDs')
    counts=pd.to_numeric(manual.manual_rep_count,errors='raise')
    if counts.isna().any() or (counts<0).any() or (counts%1!=0).any(): raise ValueError('Fill all manual counts first')
    manifest=json.loads((a.work/'manifest.json').read_text()); policy=json.loads((a.work/'mv7_exclusion_policy.json').read_text())
    vids={v['sample_id']:v for v in manifest['videos']}; excluded=set(policy['excluded_sample_ids'])
    a.out_dir.mkdir(parents=True,exist_ok=False)
    report=[]; all_events=[]
    for r in manual.itertuples(index=False):
        sid=r.sample_id; v=vids[sid]; fps=float(v['fps']); inv=inventory.loc[inventory.sample_id==sid].iloc[0]
        if sid in excluded or v['dataset']!='mv7' or v['exercise']!='push_up' or v['camera']!='side': raise ValueError(f'Unexpected sample {sid}')
        if pre.check_cache(v,manifest['weights_sha256'])!='complete': raise ValueError(f'Invalid cache {sid}')
        paths=pre.cache_paths(v); raw=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r')
        with np.load(paths[2],allow_pickle=False) as pack: status=np.asarray(pack['status'])
        if len(raw)!=int(v['frames']) or len(status)!=len(raw) or len(raw)!=int(inv.frames) or not np.isclose(fps,inv.fps) or not np.array_equal(raw['frame_id'],np.arange(len(raw))): raise ValueError(f'Frame mismatch {sid}')
        # Reference diagnostic: causal counter on the previously full-clip normalized signal.
        diag=pd.read_csv(a.pilot_dir/f'{sid}.csv')
        if not np.array_equal(diag.frame_id,np.arange(len(raw))): raise ValueError(f'Diagnostic alignment mismatch {sid}')
        old=DepthCounter(fps,a.depth,a.recovery)
        for i,x in enumerate(diag.shoulder_y_mean_norm): old.update(x,i)
        # Actual candidate: causal scale and causal median, no clip-level information.
        c=PushupCounter(fps,a.depth,a.recovery); trace=[]
        for i,item in enumerate(raw):
            event=c.update(item['kp'],i,selected=(status[i]==0 and bool(item['pose_valid'])))
            trace.append(dict(frame_id=i,t_sec=i/fps,depth_norm=c.value,filtered_depth=c.counter.filtered,torso_scale_px=c.last_scale,count=len(c.counter.events),emitted=event is not None))
        df=pd.DataFrame(trace); df.to_csv(a.out_dir/f'{sid}_trace.csv',index=False)
        events=c.counter.events
        for e in events: all_events.append(dict(sample_id=sid,**e))
        n=len(events); gt=int(r.manual_rep_count)
        report.append(dict(sample_id=sid,fps=fps,frames=len(raw),manual_count=gt,diagnostic_count=len(old.events),causal_count=n,signed_error=n-gt,absolute_error=abs(n-gt),valid_signal_pct=100*df.depth_norm.notna().mean()))
        fig,ax=plt.subplots(figsize=(12,4)); ax.plot(df.frame_id,df.depth_norm,alpha=.35,label='causal normalized depth'); ax.plot(df.frame_id,df.filtered_depth,label='trailing median')
        for e in events:
            ax.axvspan(e['start_frame'],e['end_frame'],alpha=.09,color='gray'); ax.axvline(e['end_frame'],alpha=.5,color='gray'); ax.text(e['end_frame'],.98,str(e['rep']),transform=ax.get_xaxis_transform(),fontsize=8,va='top')
        ax.set(xlabel='Frame ID',ylabel='Normalized shoulder height',title=f'{sid} | manual {gt} | causal {n}'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(a.out_dir/f'{sid}.png',dpi=120); plt.close(fig)
        print(report[-1],flush=True)
    df=pd.DataFrame(report); df.to_csv(a.out_dir/'per_video.csv',index=False)
    pd.DataFrame(all_events,columns=['sample_id','rep','start_frame','bottom_frame','end_frame','depth','duration_sec']).to_csv(a.out_dir/'candidate_reps.csv',index=False)
    summary=dict(videos=len(df),manual_total=int(df.manual_count.sum()),diagnostic_total=int(df.diagnostic_count.sum()),causal_total=int(df.causal_count.sum()),exact_count_videos=int((df.signed_error==0).sum()),mean_absolute_count_error=float(df.absolute_error.mean()),total_absolute_error=int(df.absolute_error.sum()),config=dict(depth=a.depth,recovery=a.recovery,scale_history_sec=1.5,median_sec=.2),manual_counts_sha256=hashlib.sha256((a.pilot_dir/'manual_counts.csv').read_bytes()).hexdigest(),notes='Pilot-tuned, no exercise gating. Manual clip counts only, not matched rep recall. Candidate boundaries are proposals, not GT. No correctness labels inferred.')
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    with ZipFile(a.out_dir/'counter_pilot_results.zip','w',ZIP_DEFLATED) as z:
        for f in sorted(a.out_dir.iterdir()):
            if f.suffix in ('.csv','.json','.png'): z.write(f,f.name)

if __name__=='__main__': main()
