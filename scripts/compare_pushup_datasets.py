#!/usr/bin/env python3
"""Fixed LR80 dataset comparison. No new inference or final deployment model."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_pushup_posture as base

def load(mv,kg):
    a=base.read(mv/'summary.json'); b=base.read(kg/'summary.json')
    for s in (a,b):
        if s['feature_names']!=base.FEATURE_NAMES or s['feature_source_sha256']!=base.sha(base.__file__):
            raise ValueError('Feature source/schema mismatch')
    for root,s,files in [(mv,a,[('features.npy','features_sha256'),('repetitions.csv','repetitions_sha256')]),(kg,b,[('features.npy','features_sha256'),('candidate_reps.csv','candidate_reps_sha256')])]:
        for file,key in files:
            if base.sha(root/file)!=s[key]: raise ValueError(f'Changed file {root/file}')
    d=pd.read_csv(mv/'repetitions.csv').sort_values('feature_row').reset_index(drop=True)
    k=pd.read_csv(kg/'candidate_reps.csv')
    if not set(k.accepted.unique())<={True,False}: raise ValueError('Invalid accepted flags')
    k=k[k.accepted].sort_values('feature_row').reset_index(drop=True).rename(columns={'clip_label':'label'})
    cols=['sample_id','group','label','rep','feature_row']
    d=d[cols].assign(dataset='MV7'); k=k[cols].assign(dataset='Kaggle')
    frames=[]; arrays=[]; clips=[]
    for data,root,n,nclip,ng in [(d,mv,255,23,13),(k,kg,89,86,18)]:
        X=np.load(root/'features.npy',allow_pickle=False)
        if X.shape!=(n,80) or len(data)!=n or not np.array_equal(data.feature_row,np.arange(n)) or data.sample_id.nunique()!=nclip or data.group.nunique()!=ng:
            raise ValueError('Unexpected feature inventory')
        if np.isinf(X).any() or not np.isfinite(X).any(axis=1).all() or data.duplicated(['sample_id','rep']).any():
            raise ValueError('Bad feature data')
        if not data.groupby('sample_id')[['group','label']].nunique().eq(1).all().all(): raise ValueError('Inconsistent metadata')
        dataset=data.dataset.iloc[0]
        v=pd.read_csv(root/'per_video.csv')
        if dataset=='Kaggle': v=v.rename(columns={'clip_label':'label'})
        v=v[['sample_id','group','label']].assign(dataset=dataset)
        if len(v)!=(23 if dataset=='MV7' else 100) or v.sample_id.duplicated().any(): raise ValueError('Invalid clip inventory')
        check=data.merge(v,on=['sample_id','group','label','dataset'],validate='many_to_one')
        if len(check)!=len(data): raise ValueError('Missing clip metadata')
        clips.append(v); frames.append(data); arrays.append(X)
    data=pd.concat(frames,ignore_index=True); X=np.concatenate(arrays); v=pd.concat(clips,ignore_index=True)
    if v.sample_id.duplicated().any() or set(d.group)&set(k.group): raise ValueError('Overlapping IDs')
    ref=pd.read_csv(mv/'predictions.csv').sort_values('feature_row').reset_index(drop=True)
    if not ref[cols].equals(d[cols]): raise ValueError('Reference metadata mismatch')
    return data,X,v,ref

def weights(d):
    # Equal loss mass per dataset, then class, then clip. Mean weight = 1.
    w=np.zeros(len(d)); nd=d.dataset.nunique()
    for ds in d.dataset.unique():
        for label in (0,1):
            mask=(d.dataset==ds)&(d.label==label)
            count=d.loc[mask,'sample_id'].value_counts()
            if not len(count): raise ValueError('Missing class in training domain')
            w[mask.to_numpy()]=len(d)/(nd*2*len(count)*d.loc[mask,'sample_id'].map(count).to_numpy(float))
    if not np.isclose(w.mean(),1): raise ValueError('Bad loss weights')
    return w

def run(d,X,v,ref,verify=True):
    ds=d.dataset.to_numpy(); y=d.label.to_numpy(int); g=d.group.to_numpy()
    variants=[('A_MV7_to_MV7','MV7','MV7'),('B_Kaggle_to_Kaggle','Kaggle','Kaggle'),
              ('C_Kaggle_to_MV7','Kaggle','MV7'),('D_MV7_to_Kaggle','MV7','Kaggle'),
              ('E_mixed','both','both')]
    rr=[]; cc=[]; ff=[]; mm=[]; gg=[]; pp=[]
    for variant,train_ds,test_ds in variants:
        eligible=np.ones(len(d),bool) if test_ds=='both' else ds==test_ds
        train_pool=np.ones(len(d),bool) if train_ds=='both' else ds==train_ds
        prob=np.full(len(d),np.nan); pred=np.full(len(d),-1,int)
        # Cross-domain runs fit once; in-domain/mixed leave each target group out.
        tasks=[('cross_domain',eligible,train_pool)] if train_ds!=test_ds else [
            (group,eligible&(g==group),train_pool&(g!=group)) for group in sorted(set(g[eligible]))]
        for group,te,tr in tasks:
            a,b=d.loc[tr],d.loc[te]
            if set(a.group)&set(b.group) or set(a.sample_id)&set(b.sample_id): raise ValueError('Leakage')
            w=weights(a)
            model=base.make_pipeline().fit(X[tr],y[tr],model__sample_weight=w)
            prob[te]=model.predict_proba(X[te])[:,list(model.classes_).index(1)]
            pred[te]=model.predict(X[te])
            f=dict(variant=variant,held_out_group=group,train_reps=len(a),test_reps=len(b),train_groups=a.group.nunique(),test_groups=b.group.nunique(),train_clips=a.sample_id.nunique(),test_clips=b.sample_id.nunique())
            for domain in ('MV7','Kaggle'):
                for label in (0,1):
                    f[f'{domain}_label{label}_loss_share']=float(w[((a.dataset==domain)&(a.label==label)).to_numpy()].sum()/w.sum())
            ff.append(f)
        if not np.isfinite(prob[eligible]).all() or (pred[eligible]<0).any(): raise ValueError('Incomplete predictions')
        r=d.loc[eligible].assign(variant=variant,prob_good=prob[eligible],pred=pred[eligible])
        if variant.startswith('A_') and verify:
            if not np.array_equal(r.pred.to_numpy(),ref.pred.to_numpy()) or not np.allclose(r.prob_good,ref.prob_good,atol=1e-10,rtol=0):
                raise ValueError('MV7 baseline failed to reproduce')
        c=v.copy() if test_ds=='both' else v[v.dataset==test_ds].copy()
        c=c.merge(r.groupby('sample_id').prob_good.agg(['mean','size']).reset_index(),on='sample_id',how='left',validate='one_to_one')
        c=c.rename(columns={'mean':'prob_good','size':'accepted_reps'})
        c['accepted_reps']=c.accepted_reps.fillna(0).astype(int)
        c['pred']=np.where(c.prob_good.isna(),-1,(c.prob_good>=.5).astype(int))
        c['variant']=variant
        rr.append(r); cc.append(c)
        for domain in sorted(c.dataset.unique()):
            a=r[r.dataset==domain]; b=c[c.dataset==domain]; known=b[b.pred>=0]
            for unit,sub in [('rep_weak_label',a),('clip_classified',known),('clip_all',b)]:
                report=base.metrics(sub.label,sub.pred)
                report.update(variant=variant,dataset=domain,unit=unit,coverage=(len(known)/len(b) if unit.startswith('clip') else None),group_mean_accuracy=float(sub.assign(ok=sub.label==sub.pred).groupby('group').ok.mean().mean()))
                mm.append(report)
            for group,h in b.groupby('group'):
                gg.append(dict(variant=variant,dataset=domain,group=group,**base.metrics(h.label,h.pred)))
                h=h[h.pred>=0]
                if set(h.label)=={0,1}:
                    good=h.loc[h.label==1,'prob_good'].mean(); bad=h.loc[h.label==0,'prob_good'].mean()
                    pp.append(dict(variant=variant,dataset=domain,group=group,good_mean=good,bad_mean=bad,correct_order=bool(good>bad),tie=bool(good==bad)))
        print(variant,'completed',flush=True)
    return dict(metrics=pd.DataFrame(mm),rep_predictions=pd.concat(rr,ignore_index=True),clip_predictions=pd.concat(cc,ignore_index=True),folds=pd.DataFrame(ff),per_group=pd.DataFrame(gg),paired_order=pd.DataFrame(pp))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mv7',type=Path,default=Path('results/pushup/posture_baseline'))
    p.add_argument('--kaggle',type=Path,default=Path('results/pushup/kaggle_preprocess'))
    p.add_argument('--out-dir',type=Path,default=Path('results/pushup/dataset_compare'))
    a=p.parse_args()
    if a.out_dir.exists(): raise FileExistsError(a.out_dir)
    d,X,v,ref=load(a.mv7,a.kaggle)
    with threadpool_limits(limits=1): results=run(d,X,v,ref)
    a.out_dir.mkdir(parents=True)
    for name,df in results.items(): df.to_csv(a.out_dir/(name+'.csv'),index=False)
    report=dict(metrics=json.loads(results['metrics'].to_json(orient='records')),baseline_verified=True,
                folds=len(results['folds']),notes='Fixed LR80. MV7 duplicate-linked person groups, Kaggle provisional source groups; cross-domain person independence unverified. Mixed equal dataset/class/clip loss; scaler unweighted. Clip all counts unavailable as wrong; weak rep labels; exploratory reused data; no final model.')
    (a.out_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(results['metrics'][['variant','dataset','unit','n','accuracy','balanced_accuracy','bad_recall']].to_string(index=False))
    print('Saved:',a.out_dir)
if __name__=='__main__': main()
