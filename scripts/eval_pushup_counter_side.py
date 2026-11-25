#!/usr/bin/env python3
"""Frozen causal push-up counter evaluation, 23 previously screened side clips.
Run from pose_training_offline. count mode does not show automatic predictions.
No new inference. Candidate intervals are not GT and have no rep correctness labels.
"""
from __future__ import annotations
import argparse, json, hashlib, subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd

PILOT=Path('results/pushup/pilot')
WORK=Path('data/preprocess/extra_datasets')
OUT=Path('results/pushup/side_eval')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def inventory():
    inv=pd.read_csv(PILOT/'side_inventory.csv')
    inv=inv[inv.eligible.astype(str).str.lower()=='true'].copy().sort_values('sample_id').reset_index(drop=True)
    p=pd.read_csv(PILOT/'manual_counts.csv',dtype=str).fillna('')
    if len(inv)!=23 or inv.sample_id.duplicated().any() or len(p)!=6 or p.sample_id.duplicated().any() or not set(p.sample_id)<=set(inv.sample_id):
        raise ValueError('Expected 23 eligible clips and 6 unique pilot clips')
    pg=set(inv.loc[inv.sample_id.isin(p.sample_id),'group'])
    inv['split']=np.where(inv.group.isin(pg),'pilot_group','new_group')
    if (inv.split=='pilot_group').sum()!=6 or inv.group.nunique()!=13: raise ValueError('Unexpected pilot/new group allocation')
    return inv,p

def write_counts(df,path):
    tmp=path.with_suffix('.tmp'); df.to_csv(tmp,index=False); tmp.replace(path)

def count(out):
    inv,p=inventory(); out.mkdir(parents=True,exist_ok=True)
    path=out/'manual_counts.csv'
    if path.exists(): df=pd.read_csv(path,dtype=str).fillna('')
    else:
        df=inv[['sample_id','video_path','split']].merge(p[['sample_id','manual_rep_count','notes']],on='sample_id',how='left').fillna('')
        write_counts(df,path)
    if df.sample_id.duplicated().any() or set(df.sample_id)!=set(inv.sample_id): raise ValueError('Manual count inventory mismatch')
    for i,r in df.iterrows():
        if r.manual_rep_count.isdigit(): continue
        print(f'\n[{i+1}/{len(df)}] {r.sample_id}',flush=True)
        v=inv.loc[inv.sample_id==r.sample_id].iloc[0]
        if not Path(v.video_path).is_file(): raise FileNotFoundError(v.video_path)
        w=subprocess.check_output(['wslpath','-w',v.video_path],text=True).strip()
        subprocess.Popen(['explorer.exe',w])
        while True:
            val=input('Manual count (q=save and quit, s=skip): ').strip().lower()
            if val=='q': return
            if val=='s' or val.isdigit(): break
        if val=='s': continue
        df.at[i,'manual_rep_count']=val
        df.at[i,'notes']=input('Notes (optional): ').strip()
        write_counts(df,path)
    print('Counts entered:',sum(df.manual_rep_count.str.isdigit()),'/',len(df))

def evaluate(out):
    from eval_pushup_counter_pilot import PushupCounter
    import preprocess_rehab24 as pre
    from extract_video_kp_cache import KP_DUMP_DTYPE
    inv,p=inventory(); cp=out/'manual_counts.csv'
    manual=pd.read_csv(cp,dtype=str).fillna('')
    if manual.sample_id.duplicated().any() or set(manual.sample_id)!=set(inv.sample_id) or not manual.manual_rep_count.str.isdigit().all(): raise ValueError('Complete all 23 manual counts first')
    if any(int(manual.set_index('sample_id').at[r.sample_id,'manual_rep_count'])!=int(r.manual_rep_count) for r in p.itertuples()): raise ValueError('Pilot counts changed')
    m=json.loads((WORK/'manifest.json').read_text()); pol=json.loads((WORK/'mv7_exclusion_policy.json').read_text()); vids={v['sample_id']:v for v in m['videos']}; excluded=set(pol['excluded_sample_ids']); groups=pol['person_fold_groups']
    dest=out/'evaluation'; dest.mkdir(exist_ok=False)
    rows=[]; events=[]
    for r in inv.itertuples(index=False):
        sid=r.sample_id; v=vids[sid]; fps=float(v['fps']); n=int(v['frames'])
        if sid in excluded or v['dataset']!='mv7' or v['exercise']!='push_up' or v['camera']!='side' or 'mv7:'+groups[v['person_id']]!=r.group or int(v['clip_correctness'])!=int(r.clip_label): raise ValueError(f'Metadata mismatch {sid}')
        if n!=int(r.frames) or not np.isclose(fps,r.fps) or pre.check_cache(v,m['weights_sha256'])!='complete': raise ValueError(f'Cache mismatch {sid}')
        paths=pre.cache_paths(v); raw=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r')
        with np.load(paths[2],allow_pickle=False) as pack: status=pack['status']; boxes=pack['selected_box']
        if len(raw)!=n or len(status)!=n or len(boxes)!=n or not np.array_equal(raw['frame_id'],np.arange(n)): raise ValueError(f'Frame mismatch {sid}')
        selected=status==0; k=raw['kp']; ok=np.isfinite(k).all(2)&(k[:,:,2]>=.3)&selected[:,None]
        chain=ok[:,[5,7,9,11,13,15]].all(1)|ok[:,[6,8,10,12,14,16]].all(1)
        edges=np.flatnonzero(np.diff(np.r_[False,~chain,False])); gaps=edges[1::2]-edges[::2]
        center=(boxes[:,:2]+boxes[:,2:])/2; diag=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
        jump=selected[:-1]&selected[1:]&(np.linalg.norm(np.diff(center,axis=0),axis=1)/diag[:-1]>=.25)
        if selected.mean()<.95 or chain.mean()<.90 or max(gaps,default=0)/fps>.5 or jump.any(): raise ValueError(f'Quality changed {sid}')
        c=PushupCounter(fps,depth=.18,recovery=.75); trace=[]
        for i,item in enumerate(raw):
            e=c.update(item['kp'],i,selected=bool(selected[i] and item['pose_valid']))
            trace.append(dict(frame_id=i,t_sec=i/fps,depth_norm=c.value,filtered_depth=c.counter.filtered,torso_scale_px=c.last_scale,count=len(c.counter.events),emitted=e is not None))
        t=pd.DataFrame(trace); t.to_csv(dest/f'{sid}_trace.csv',index=False)
        for e in c.counter.events: events.append(dict(sample_id=sid,group=r.group,split=r.split,clip_label=int(r.clip_label),**e))
        gt=int(manual.set_index('sample_id').at[sid,'manual_rep_count']); pred=len(c.counter.events)
        rows.append(dict(sample_id=sid,group=r.group,split=r.split,clip_label=int(r.clip_label),fps=fps,frames=n,manual_count=gt,causal_count=pred,signed_error=pred-gt,absolute_error=abs(pred-gt),valid_signal_pct=100*t.depth_norm.notna().mean()))
        print(sid, 'manual',gt,'auto',pred,'error',pred-gt,flush=True)
    df=pd.DataFrame(rows); df.to_csv(dest/'per_video.csv',index=False)
    ev=pd.DataFrame(events,columns=['sample_id','group','split','clip_label','rep','start_frame','bottom_frame','end_frame','depth','duration_sec']); ev.to_csv(dest/'candidate_reps.csv',index=False)
    def metrics(d):
        return dict(videos=len(d),groups=d.group.nunique(),manual_total=int(d.manual_count.sum()),causal_total=int(d.causal_count.sum()),exact_count_videos=int((d.signed_error==0).sum()),mean_absolute_count_error=float(d.absolute_error.mean()),total_absolute_error=int(d.absolute_error.sum()),undercount_total=int((-d.signed_error.clip(upper=0)).sum()),overcount_total=int(d.signed_error.clip(lower=0).sum()))
    pilot=df[df.split=='pilot_group'];
    if dict(zip(pilot.sample_id,pilot.causal_count)) != dict(zip(pilot.sample_id,pilot.sample_id.map({'MV7_subject_001_push_up_good_side':12,'MV7_subject_001_push_up_bad_side':17,'MV7_subject_002_push_up_good_side':12,'MV7_subject_002_push_up_bad_side':14,'MV7_subject_003_push_up_good_side':18,'MV7_subject_003_push_up_bad_side':14}))): raise RuntimeError('Frozen pilot reproduction failed')
    report={'all':metrics(df),'pilot_group':metrics(pilot),'new_group':metrics(df[df.split=='new_group']),'new_group_by_clip_label':{str(l):metrics(d) for l,d in df[df.split=='new_group'].groupby('clip_label')},'config':{'depth':.18,'recovery':.75,'scale_history_sec':1.5,'median_sec':.2},'manual_counts_sha256':sha(cp),'counter_source_sha256':sha(Path(__file__).with_name('eval_pushup_counter_pilot.py')),'notes':'Frozen counter, no exercise gating. New groups exclude all pilot groups. Clip count errors only, not matched recall. Candidate intervals are NOT GT; clip_label is NOT a repetition label.'}
    (dest/'summary.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    with ZipFile(dest/'side_results.zip','w',ZIP_DEFLATED) as z:
        for f in sorted(dest.iterdir()):
            if f.suffix in ('.csv','.json'): z.write(f,f.name)
        z.write(cp,'manual_counts.csv')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['count','evaluate']); p.add_argument('--out-dir',type=Path,default=OUT); a=p.parse_args()
    (count if a.mode=='count' else evaluate)(a.out_dir)
if __name__=='__main__': main()
