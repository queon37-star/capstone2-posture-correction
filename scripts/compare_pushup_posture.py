#!/usr/bin/env python3
"""Fixed exploratory comparisons using the existing side-view push-up features.
No keypoint extraction, counter tuning, threshold tuning, or deployment model.
Run beside train_pushup_posture.py from pose_training_offline.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits
import train_pushup_posture as base

COMPACT = ['elbow_angle_p10','elbow_angle_p90','elbow_angle_p90_p10',
           'shoulder_ankle_y_p90_p10','hip_alignment_p10','hip_alignment_median',
           'hip_offset_abs_p90','knee_angle_p10','shoulder_wrist_x_abs_median',
           'shoulder_wrist_y_p90_p10','torso_tilt_median','elbow_asymmetry_p90']


def oof(d, X, estimator):
    y=d.label.to_numpy(int); prob=np.full(len(d),np.nan); pred=np.full(len(d),-1,int); folds=[]
    for tr,te in LeaveOneGroupOut().split(X,y,d.group):
        a,b=d.iloc[tr],d.iloc[te]
        for c in ('group','person_id','sample_id'):
            if set(a[c])&set(b[c]): raise ValueError(f'Leakage: {c}')
        w=base.loss_weights(a)
        model=clone(estimator).fit(X[tr],y[tr],model__sample_weight=w)
        prob[te]=model.predict_proba(X[te])[:,list(model.classes_).index(1)]
        pred[te]=model.predict(X[te])
        folds.append(dict(group=str(b.group.iloc[0]),train_rows=len(tr),test_rows=len(te),train_clips=a.sample_id.nunique(),test_clips=b.sample_id.nunique(),**base.metrics(y[te],pred[te])))
    if not np.isfinite(prob).all() or (pred<0).any(): raise ValueError('Incomplete OOF')
    return prob,pred,folds


def compare(d,X,reference):
    if X.shape!=(255,80) or len(d)!=255 or d.sample_id.nunique()!=23 or d.group.nunique()!=13: raise ValueError('Unexpected inventory')
    if d.duplicated(['sample_id','rep']).any() or not np.array_equal(d.feature_row,np.arange(len(d))): raise ValueError('Bad row mapping')
    if np.isinf(X).any() or not np.isfinite(X).any(axis=1).all(): raise ValueError('Bad features')
    if not d.groupby('sample_id')[['person_id','group','label']].nunique().eq(1).all().all(): raise ValueError('Clip metadata inconsistent')
    ix=[base.FEATURE_NAMES.index(n) for n in COMPACT]; compact=X[:,ix]
    clips=d[['sample_id','person_id','group','split','label']].drop_duplicates().reset_index(drop=True)
    if len(clips)!=23: raise ValueError('Bad clip mapping')
    # Aggregate only the rows of the clip being predicted. No cross-clip statistics.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        clipX=np.stack([np.nanmedian(compact[d.sample_id.eq(s).to_numpy()],axis=0) for s in clips.sample_id])
    rf=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('model',RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=5,random_state=42,n_jobs=1))])
    variants=[('LR80_reference',d,X,base.make_pipeline(),False),('LR12_rep',d,compact,base.make_pipeline(),False),('RF80_rep',d,X,rf,False),('LR12_clip',clips,clipX,base.make_pipeline(),True)]
    reports=[]; rep_rows=[]; clip_rows=[]; fold_rows=[]; pair_rows=[]
    for name,dd,xx,est,isclip in variants:
        prob,pred,folds=oof(dd,xx,est)
        fold_rows.extend([dict(variant=name,unit='clip' if isclip else 'rep',**r) for r in folds])
        if name=='LR80_reference':
            if not np.array_equal(pred,reference.pred.to_numpy(int)) or not np.allclose(prob,reference.prob_good.to_numpy(float),rtol=0,atol=1e-10): raise RuntimeError('Baseline replay mismatch; stop comparison')
        if isclip:
            c=clips.assign(prob_good=prob,pred=pred)
        else:
            r=d.assign(prob_good=prob,pred=pred,variant=name)
            rep_rows.append(r)
            c=clips.merge(r.groupby('sample_id',as_index=False).prob_good.mean(),on='sample_id',validate='one_to_one')
            c['pred']=(c.prob_good>=.5).astype(int)
            reports.append(dict(variant=name,unit='rep_weak_label',**base.metrics(r.label,r.pred)))
        reports.append(dict(variant=name,unit='clip',**base.metrics(c.label,c.pred)))
        c=c.assign(variant=name); clip_rows.append(c)
        for person,g in c.groupby('person_id'):
            if len(g)==2 and set(g.label)=={0,1}:
                good=float(g.loc[g.label==1,'prob_good'].iloc[0]); bad=float(g.loc[g.label==0,'prob_good'].iloc[0]); pair_rows.append(dict(variant=name,person_id=person,group=g.group.iloc[0],prob_good_on_good=good,prob_good_on_bad=bad,correct_order=good>bad,tie=good==bad))
        print(name,base.metrics(c.label,c.pred),flush=True)
    return pd.DataFrame(reports),pd.concat(rep_rows,ignore_index=True),pd.concat(clip_rows,ignore_index=True),pd.DataFrame(fold_rows),pd.DataFrame(pair_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,default=base.OUT);p.add_argument('--out-dir',type=Path,default=Path('results/pushup/posture_compare'));a=p.parse_args()
    if a.out_dir.exists(): raise FileExistsError(a.out_dir)
    s=base.read(a.baseline/'summary.json')
    for file,key in [('features.npy','features_sha256'),('repetitions.csv','repetitions_sha256')]:
        if base.sha(a.baseline/file)!=s[key]: raise ValueError(f'Changed file: {file}')
    if base.sha(base.__file__)!=s['feature_source_sha256'] or s['feature_names']!=base.FEATURE_NAMES: raise ValueError('Changed feature source/schema')
    d=pd.read_csv(a.baseline/'repetitions.csv'); X=np.load(a.baseline/'features.npy',allow_pickle=False)
    ref=pd.read_csv(a.baseline/'predictions.csv').sort_values('feature_row').reset_index(drop=True)
    if not d[['sample_id','rep','label','group','feature_row']].equals(ref[['sample_id','rep','label','group','feature_row']]): raise ValueError('Reference metadata mismatch')
    with threadpool_limits(limits=1): m,r,c,f,pairs=compare(d,X,ref)
    a.out_dir.mkdir(parents=True)
    for df,name in [(m,'metrics'),(r,'rep_predictions'),(c,'clip_predictions'),(f,'folds'),(pairs,'paired_order')]: df.to_csv(a.out_dir/(name+'.csv'),index=False)
    report=dict(metrics=json.loads(m.to_json(orient='records')),paired_order=json.loads(pairs.groupby('variant').agg(pairs=('correct_order','size'),correct=('correct_order','sum'),ties=('tie','sum')).reset_index().to_json(orient='records')),compact_features=COMPACT,baseline_verified=True,features_sha256=s['features_sha256'],notes='Same exploratory groups and weak labels; LR12_clip predicts a whole clip from median rep features, not individual reps. No threshold search or final model. RF is a fixed comparison, not tuned.')
    (a.out_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');print(m.to_string(index=False));print('Saved:',a.out_dir)

if __name__=='__main__': main()
