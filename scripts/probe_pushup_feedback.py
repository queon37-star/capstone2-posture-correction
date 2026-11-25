#!/usr/bin/env python3
"""Exploratory push-up geometry feedback. Not a validated posture classifier.
Uses frozen MV7/Kaggle 80-D features, no training or threshold search.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLDS = dict(hip_angle_p10_deg=160.0, hip_offset_abs_p90_torso=0.20,
                  elbow_angle_p10_deg=110.0, shoulder_range_torso=0.30)
FEATURES = ('hip_alignment_p10','hip_offset_abs_p90','elbow_angle_p10',
            'shoulder_ankle_y_p90_p10')

def evaluate(data, X, clips, names):
    if X.shape != (len(data),len(names)):
        raise ValueError('Feature shape mismatch')
    r = data.copy()
    for name in FEATURES:
        r[name] = X[:,names.index(name)]
    h = r.hip_alignment_p10.to_numpy()
    o = r.hip_offset_abs_p90.to_numpy()
    e = r.elbow_angle_p10.to_numpy()
    s = r.shoulder_ankle_y_p90_p10.to_numpy()
    hv = np.isfinite(h) & np.isfinite(o)
    dv = np.isfinite(e) & np.isfinite(s)
    # 1: candidate warning; 0: criterion not triggered; -1: not assessable.
    r['alignment_flag'] = np.where(hv,((h<THRESHOLDS['hip_angle_p10_deg']) &
            (o>THRESHOLDS['hip_offset_abs_p90_torso'])).astype(int),-1)
    r['depth_flag'] = np.where(dv,((e>THRESHOLDS['elbow_angle_p10_deg']) &
            (s<THRESHOLDS['shoulder_range_torso'])).astype(int),-1)
    a,b = r.alignment_flag.to_numpy(),r.depth_flag.to_numpy()
    r['any_flag'] = np.where((a==1)|(b==1),1,np.where((a==0)&(b==0),0,-1))
    r['feedback'] = np.select([(a==1)&(b==1),a==1,b==1,r.any_flag==-1],
                 ['alignment_and_depth','alignment','depth','unassessable'],
                 default='no_flag')
    fields = ('alignment_flag','depth_flag','any_flag')
    rows=[]
    for sid, sub in r.groupby('sample_id',sort=False):
        item={'sample_id':sid,'accepted_reps':len(sub)}
        for field in fields:
            valid=sub[field]>=0
            item[field+'_assessable']=int(valid.sum())
            item[field+'_count']=int((sub[field]==1).sum())
            item[field+'_rate']=float((sub.loc[valid,field]==1).mean()) if valid.any() else np.nan
        rows.append(item)
    c=clips.merge(pd.DataFrame(rows),on='sample_id',how='left',validate='one_to_one')
    for col in ['accepted_reps']+[f+'_'+suffix for f in fields for suffix in ('assessable','count')]:
        c[col]=c[col].fillna(0).astype(int)
    metrics=[]
    for domain in sorted(c.dataset.unique()):
        for label in (1,0):
            u=r[(r.dataset==domain)&(r.label==label)]
            v=c[(c.dataset==domain)&(c.label==label)]
            for field in fields:
                assessed=u[field]>=0
                rate=v[field+'_rate'].dropna()
                metrics.append(dict(dataset=domain,clip_label=label,rule=field,
                      rep_total=len(u),rep_assessable=int(assessed.sum()),
                      rep_flagged=int((u[field]==1).sum()),
                      rep_flag_rate=float((u.loc[assessed,field]==1).mean()) if assessed.any() else np.nan,
                      clips_total=len(v),clips_assessable=len(rate),
                      mean_clip_flag_rate=float(rate.mean()) if len(rate) else np.nan,
                      clips_with_any_flag=int((rate>0).sum())))
    paired=[]
    for (domain,group),sub in c.groupby(['dataset','group']):
        for field in fields:
            val=field+'_rate'
            good=sub.loc[sub.label==1,val].dropna()
            bad=sub.loc[sub.label==0,val].dropna()
            if len(good) and len(bad):
                g=float(good.mean()); b=float(bad.mean())
                paired.append(dict(dataset=domain,group=group,rule=field,
                         good_clip_mean_flag_rate=g,bad_clip_mean_flag_rate=b,
                         bad_higher=b>g,tie=b==g))
    return r,c,pd.DataFrame(metrics),pd.DataFrame(paired)

def main():
    import compare_pushup_datasets as source
    import train_pushup_posture as base
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mv7',type=Path,default=Path('results/pushup/posture_baseline'))
    p.add_argument('--kaggle',type=Path,default=Path('results/pushup/kaggle_preprocess'))
    p.add_argument('--out-dir',type=Path,default=Path('results/pushup/feedback_probe'))
    a=p.parse_args()
    if a.out_dir.exists(): raise FileExistsError(a.out_dir)
    d,X,v,_=source.load(a.mv7,a.kaggle)
    r,c,m,q=evaluate(d,X,v,base.FEATURE_NAMES)
    a.out_dir.mkdir(parents=True)
    for name,frame in [('rep_feedback',r),('per_video',c),('metrics',m),('paired_order',q)]:
        frame.to_csv(a.out_dir/(name+'.csv'),index=False)
    report=dict(thresholds=THRESHOLDS,rep_count=len(r),clip_count=len(c),
                metrics=json.loads(m.to_json(orient='records')),
                paired_order=json.loads(q.groupby(['dataset','rule']).agg(
                  groups=('group','size'),bad_higher=('bad_higher','sum'),ties=('tie','sum')
                ).reset_index().to_json(orient='records')) if len(q) else [],
                notes='Fixed exploratory engineering criteria, not clinical limits. 2D side-view only. Original labels describe clips, not individual errors. Flag rates are descriptive; no flag does not establish correct posture. No training, tuning or validation of error-specific accuracy.')
    (a.out_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(m.to_string(index=False))
    print('Saved:',a.out_dir)

if __name__=='__main__':
    main()
