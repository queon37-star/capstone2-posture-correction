#!/usr/bin/env python3
"""Side-view MV7 push-up posture LR. Labels inherited from clips, NOT rep GT.
Run from pose_training_offline. Read only existing cache and counter outputs.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from threadpoolctl import threadpool_limits

WORK=Path('data/preprocess/extra_datasets')
SIDE=Path('results/pushup/side_eval/evaluation')
OUT=Path('results/pushup/posture_baseline')
SIGNALS=('elbow_angle','shoulder_ankle_y','hip_alignment','hip_offset','hip_offset_abs','knee_angle','shoulder_wrist_x_abs','shoulder_wrist_y','torso_tilt','elbow_asymmetry')
STATS=('mean','std','p10','median','p90','p90_p10','first','last')
FEATURE_NAMES=[f'{s}_{t}' for s in SIGNALS for t in STATS]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
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

def build(work,side):
    import preprocess_rehab24 as pre
    from extract_video_kp_cache import KP_DUMP_DTYPE
    m=read(work/'manifest.json'); pol=read(work/'mv7_exclusion_policy.json'); ex=set(pol['excluded_sample_ids']); groups=pol['person_fold_groups']
    if len(ex)!=67: raise ValueError('Duplicate exclusion policy changed')
    summary=read(side/'summary.json'); cp=Path(__file__).with_name('eval_pushup_counter_pilot.py')
    if sha(cp)!=summary['counter_source_sha256']: raise ValueError('Counter source changed after evaluation')
    if summary['all']['causal_total']!=258 or summary['all']['videos']!=23: raise ValueError('Expected frozen side evaluation')
    vids={v['sample_id']:v for v in m['videos']}; reps=pd.read_csv(side/'candidate_reps.csv'); pv=pd.read_csv(side/'per_video.csv')
    if len(reps)!=258 or len(pv)!=23 or pv.sample_id.duplicated().any() or reps.duplicated(['sample_id','rep']).any() or set(reps.sample_id)!=set(pv.sample_id): raise ValueError('Unexpected candidate inventory')
    if pv.group.nunique()!=13 or set(reps.clip_label)!={0,1}: raise ValueError('Unexpected groups/labels')
    rows=[]; feats=[]; accepted=[]
    for r in pv.itertuples(index=False):
        sid=r.sample_id; v=vids[sid]; fps=float(v['fps']); n=int(v['frames'])
        if sid in ex or (v['dataset'],v['exercise'],v['camera'])!=('mv7','push_up','side') or 'mv7:'+groups[v['person_id']]!=r.group or int(v['clip_correctness'])!=r.clip_label: raise ValueError(f'metadata mismatch {sid}')
        if n!=r.frames or not np.isclose(fps,r.fps) or pre.check_cache(v,m['weights_sha256'])!='complete': raise ValueError(f'cache mismatch {sid}')
        paths=pre.cache_paths(v); raw=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r')
        with np.load(paths[2],allow_pickle=False) as z: status=z['status']; boxes=z['selected_box']
        if len(raw)!=n or len(status)!=n or len(boxes)!=n or not np.array_equal(raw['frame_id'],np.arange(n)): raise ValueError(f'frame mismatch {sid}')
        selected=(status==0)&(raw['pose_valid']!=0)
        center=(boxes[:,:2]+boxes[:,2:])/2; diag=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
        jump=(status[:-1]==0)&(status[1:]==0)&(np.linalg.norm(np.diff(center,axis=0),axis=1)/diag[:-1]>=.25)
        sub=reps[reps.sample_id==sid].sort_values('rep')
        if len(sub)!=r.causal_count or list(sub.rep)!=list(range(1,len(sub)+1)): raise ValueError(f'candidate count mismatch {sid}')
        prev=-1
        for e in sub.itertuples(index=False):
            st,bt,ed=map(int,[e.start_frame,e.bottom_frame,e.end_frame])
            if not 0<=st<bt<ed<n or st<prev or e.group!=r.group or e.clip_label!=r.clip_label or e.split!=r.split: raise ValueError(f'bad interval/metadata {sid}/{e.rep}')
            prev=ed
            k=np.array(raw['kp'][st:ed+1],dtype=np.float32,copy=True); sel=selected[st:ed+1]
            q,why=quality(k,sel,fps)
            if jump[max(0,st-1):min(len(jump),ed+1)].any(): why.append('box_jump_touching_rep')
            k[~sel,:,:2]=np.nan; k[~sel,:,2]=0
            x=None
            if not why:
                try: x=extract_features(k)
                except ValueError as err: why.append(f'feature_failure:{err}')
            row=dict(sample_id=sid,person_id=v['person_id'],group=r.group,split=r.split,rep=int(e.rep),start_frame=st,bottom_frame=bt,end_frame=ed,fps=fps,label=int(r.clip_label),label_scope='inherited_clip',manual_count=int(r.manual_count),causal_count=int(r.causal_count),**q,accepted=not why,reason=';'.join(why))
            rows.append(row)
            if not why:
                accepted.append({**row,'feature_row':len(feats),'finite_features':int(np.isfinite(x).sum())}); feats.append(x)
        print(f'{sid}: {sum(x["sample_id"]==sid for x in accepted)}/{len(sub)} accepted',flush=True)
    if not feats: raise ValueError('No usable repetitions')
    return pd.DataFrame(accepted),np.stack(feats),pd.DataFrame(rows),pv,m['weights_sha256']

def make_pipeline():
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler()),('model',LogisticRegression(C=.1,class_weight=None,max_iter=3000,random_state=42))])

def loss_weights(d):
    # Mean one. Each class receives half the mass, every clip within a class equal.
    y=d.label.to_numpy(); w=np.zeros(len(d),float)
    for label in (0,1):
        sel=y==label; counts=d.loc[sel,'sample_id'].value_counts()
        if not len(counts): raise ValueError('training fold lacks a class')
        w[sel]=len(d)/(2*len(counts)*d.loc[sel,'sample_id'].map(counts).to_numpy(float))
    if not np.isclose(w.mean(),1): raise AssertionError('invalid weights')
    return w

def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int)
    gg=int(((y==1)&(p==1)).sum()); gb=int(((y==1)&(p==0)).sum()); bg=int(((y==0)&(p==1)).sum()); bb=int(((y==0)&(p==0)).sum())
    def div(a,b): return float(a/b) if b else None
    gr=div(gg,int((y==1).sum())); br=div(bb,int((y==0).sum()))
    return dict(n=len(y),good=int((y==1).sum()),bad=int((y==0).sum()),correct=gg+bb,accuracy=div(gg+bb,len(y)),good_recall=gr,bad_recall=br,bad_precision=div(bb,bb+gb),balanced_accuracy=(gr+br)/2 if gr is not None and br is not None else None,good_as_good=gg,good_as_bad=gb,bad_as_good=bg,bad_as_bad=bb,unclassified=int((p<0).sum()))

def evaluate(d,X,pv,out,provenance):
    if len(d)!=len(X) or X.shape[1]!=80 or not np.array_equal(d.feature_row,np.arange(len(d))): raise ValueError('feature index mismatch')
    if d.group.nunique()<3 or set(d.label)!={0,1}: raise ValueError('Insufficient data')
    d=d.copy(); y=d.label.to_numpy(int); p=np.full(len(d),-1); prob=np.full(len(d),np.nan); folds=[]
    models=out/'models_lopo'; models.mkdir()
    for tr,te in LeaveOneGroupOut().split(X,y,d.group):
        train=d.iloc[tr]; test=d.iloc[te]; group=str(test.group.iloc[0]); tg=sorted(train.group.unique().tolist())
        if group in tg or set(train.sample_id)&set(test.sample_id) or set(train.person_id)&set(test.person_id): raise AssertionError('LOPO leakage')
        w=loss_weights(train); model=clone(make_pipeline()).fit(X[tr],y[tr],model__sample_weight=w)
        p[te]=model.predict(X[te]); prob[te]=model.predict_proba(X[te])[:,list(model.classes_).index(1)]
        artifact=dict(pipeline=model,feature_names=FEATURE_NAMES,feature_variant='Pushup80',training_variant='Pushup80_clip_balanced_lopo',label_map={0:'bad',1:'good'},label_scope='inherited_clip',held_out_group=group,training_groups=tg,training_person_ids=sorted(train.person_id.unique().tolist()),training_sample_ids=sorted(train.sample_id.unique().tolist()),training_repetitions=len(tr),training_clips=train.sample_id.nunique(),**provenance)
        path=models/(group.replace(':','_')+'.joblib'); joblib.dump(artifact,path)
        loaded=joblib.load(path); diff=float(np.max(np.abs(loaded['pipeline'].predict_proba(X[te])[:,1]-prob[te])))
        if diff!=0: raise RuntimeError('reload changed probabilities')
        folds.append(dict(group=group,train_reps=len(tr),train_clips=train.sample_id.nunique(),train_groups=len(tg),test_reps=len(te),test_clips=test.sample_id.nunique(),max_abs_reload_diff=diff,train_bad_weight=float(w[y[tr]==0].sum()),train_good_weight=float(w[y[tr]==1].sum()),**metrics(y[te],p[te])))
    if (p<0).any() or not np.isfinite(prob).all(): raise AssertionError('missing OOF predictions')
    d['pred']=p; d['prob_good']=prob; d.to_csv(out/'predictions.csv',index=False); pd.DataFrame(folds).to_csv(out/'folds.csv',index=False)
    clips=[]
    for r in pv.itertuples(index=False):
        s=d[d.sample_id==r.sample_id]; mean=float(s.prob_good.mean()) if len(s) else None
        pred=int(mean>=.5) if mean is not None else -1
        clips.append(dict(sample_id=r.sample_id,group=r.group,split=r.split,label=int(r.clip_label),manual_count=int(r.manual_count),candidate_reps=int(r.causal_count),accepted_reps=len(s),prob_good_mean=mean,pred=pred,**{f'rep_{k}':v for k,v in metrics(s.label,s.pred).items()}))
    c=pd.DataFrame(clips); c.to_csv(out/'per_video.csv',index=False)
    # Missing clips are explicitly counted as wrong for the all-clips metric.
    rep_report=metrics(d.label,d.pred); clip_report=metrics(c.label,c.pred); known=c[c.pred>=0]
    rep_report['candidate_total']=int(pv.causal_count.sum()); rep_report['coverage']=len(d)/int(pv.causal_count.sum())
    clip_report['classified_clips']=len(known); clip_report['coverage']=len(known)/len(c)
    report=dict(repetition_weak_label=rep_report,clip_all=clip_report,clip_classified=metrics(known.label,known.pred),group_mean_rep_accuracy=float(d.assign(correct=d.label==d.pred).groupby('group').correct.mean().mean()),group_mean_clip_accuracy=float(c.assign(correct=c.label==c.pred).groupby('group').correct.mean().mean()),count_matched_clips_only=dict(repetition_weak_label=metrics(d.loc[d.manual_count==d.causal_count,'label'],d.loc[d.manual_count==d.causal_count,'pred']),clip=metrics(c.loc[c.manual_count==c.candidate_reps,'label'],c.loc[c.manual_count==c.candidate_reps,'pred'])),groups=int(d.group.nunique()),models_saved=len(folds),notes='Exploratory LOPO; labels inherited from clips, candidate boundaries not GT. No repeat-level ground truth. Clip predictions average OOF rep probabilities; unavailable clips wrong. Clip-balanced and class-balanced LR loss; imputer/scaler unweighted. Side views only; no exercise gating, no independent final test, no final deployment model.')
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--work',type=Path,default=WORK); ap.add_argument('--side-dir',type=Path,default=SIDE); ap.add_argument('--out-dir',type=Path,default=OUT); a=ap.parse_args()
    if a.out_dir.exists(): raise FileExistsError(a.out_dir)
    d,X,audit,pv,weights_sha=build(a.work,a.side_dir); a.out_dir.mkdir(parents=True,exist_ok=False)
    audit.to_csv(a.out_dir/'quality_audit.csv',index=False); d.to_csv(a.out_dir/'repetitions.csv',index=False); np.save(a.out_dir/'features.npy',X)
    provenance=dict(feature_source_sha256=sha(__file__),features_sha256=sha(a.out_dir/'features.npy'),repetitions_sha256=sha(a.out_dir/'repetitions.csv'),candidate_reps_sha256=sha(a.side_dir/'candidate_reps.csv'),pose_weights_sha256=weights_sha,sklearn_version=sklearn.__version__)
    with threadpool_limits(limits=1): report=evaluate(d,X,pv,a.out_dir,provenance)
    report.update(feature_names=FEATURE_NAMES,feature_shape=list(X.shape),quality_rejections=int(len(audit)-len(d)),**provenance)
    (a.out_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Accepted by label:',d.label.value_counts().to_dict()); print(json.dumps(report,indent=2)); print('Saved:',a.out_dir)

if __name__=='__main__': main()
