#!/usr/bin/env python3
"""Initial three-class recognition experiment from existing COCO17 caches.
Run at pose_training_offline root with this file in scripts/. No raw data writes.
MV7 clip labels are weak; REHAB labels cover annotated repetitions only.
"""
import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

ROOT=Path('data/preprocess/extra_datasets')
R24=Path('data/preprocess/rehab24')
LABELS=['Squat','Pushup','Other']

def load(p): return json.loads(p.read_text())
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

def windows(v,k,status,boxes,seconds,dataset):
    n=len(k); fps=float(v['fps']); selected=status==0
    valid=np.isfinite(k[:,:,:2]).all(-1)&(k[:,:,2]>=.3)
    center=(boxes[:,:2]+boxes[:,2:])/2
    scale=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
    jump=selected[:-1]&selected[1:]&(np.linalg.norm(np.diff(center,axis=0),axis=1)/scale[:-1]>=.25)
    if dataset=='mv7':
        if selected.mean()<.95 or jump.any(): return
        if v['exercise'] not in ('squat','push_up') and (selected.mean()<.99 or valid.mean()<.75): return
        mask=np.ones(n,bool); lo=int(np.ceil(.5*fps)); hi=n-int(np.ceil(.5*fps))
    else:
        mask=np.zeros(n,bool); lo=0; hi=n
        for r in v['annotations']:
            a,b=int(r['first_frame']),int(r['last_frame'])+1
            if 0<=a<b<=n and selected[a:b].mean()>=.99: mask[a:b]=True
    span=int(round(seconds*fps)); stride=max(1,int(round(.5*fps)))
    for start in range(lo,hi-span+1,stride):
        end=start+span
        if selected[start:end].mean()<.95 or jump[start:end-1].any(): continue
        gt=mask[start:end]
        if gt.mean()<.8 or longest_bad(gt)>round(.5*fps): continue
        # REHAB windows may include short unannotated inter-repetition gaps.
        ids=start+np.floor(np.arange(round(seconds*15))*fps/15).astype(int)
        if ids[-1]>=end: continue
        data=np.array(k[ids],copy=True); data[~selected[ids],:,:]=np.nan
        x=features(data)
        if x is not None and np.isfinite(x).all(): yield start,end,x

def build(seconds,out):
    import preprocess_rehab24 as pre
    from extract_video_kp_cache import KP_DUMP_DTYPE
    r,e=load(R24/'batch_manifest.json'),load(ROOT/'manifest.json')
    if r['weights_sha256']!=e['weights_sha256']: raise ValueError('Pose model hash mismatch')
    policy=load(ROOT/'mv7_exclusion_policy.json'); excluded=set(policy['excluded_sample_ids']); groups=policy['person_fold_groups']
    if len(excluded)!=67: raise ValueError('Unexpected duplicate exclusion inventory')
    q=pd.read_csv(ROOT/'mv7_cached_body_quality.csv')
    if len(q)!=129 or q.sample_id.duplicated().any(): raise ValueError('Expected updated 129-row MV7 quality CSV')
    passed=set(q.loc[q.quality_pass.astype(str).str.lower()=='true','sample_id'])
    other=load(ROOT/'mv7_other_screening.json'); auto={x['sample_id'] for x in other['records'] if x['status'] in ('auto','existing')}
    xs=[]; rows=[]; log=[]
    for ds,vs in [('rehab24',r['videos']),('extra',e['videos'])]:
        for v in vs:
            d=ds if ds=='rehab24' else v['dataset']; sid=v['sample_id']; ex=v['exercise']; label={'squat':'Squat','push_up':'Pushup'}.get(ex,'Other')
            if d=='rehab26' and ex!='squat': continue
            if d=='rehab24' and sid in {'Ex1_Camera18_PM_109','Ex1_Camera18_PM_122'}: continue
            if d=='mv7' and (sid in excluded or sid not in (passed if label!='Other' else auto)): continue
            state=pre.check_cache(v,r['weights_sha256'])
            if state!='complete': raise RuntimeError(f'Expected cache {sid}: {state}')
            paths=pre.cache_paths(v); k=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r')['kp']
            with np.load(paths[2],allow_pickle=False) as z: status=z['status']; boxes=z['selected_box']
            group=f'{d}:{groups[v["person_id"]] if d=="mv7" else v["person_id"]}'
            count=0
            for a,b,x in windows(v,k,status,boxes,seconds,d):
                xs.append(x); rows.append(dict(sample_id=sid,dataset=d,person_id=v['person_id'],group=group,exercise=ex,camera=v['camera'],label=label,first_frame=a,last_frame=b-1)); count+=1
            log.append(dict(sample_id=sid,dataset=d,label=label,windows=count))
    if not xs: raise RuntimeError('No windows')
    df=pd.DataFrame(rows); X=np.stack(xs); np.save(out/'features.npy',X); df.to_csv(out/'windows.csv',index=False); pd.DataFrame(log).to_csv(out/'video_inventory.csv',index=False)
    print('Windows by dataset and class:\n',pd.crosstab(df.dataset,df.label)); print('Groups by dataset and class:\n',df.groupby(['dataset','label']).group.nunique()); print('Videos by dataset and class:\n',df.groupby(['dataset','label']).sample_id.nunique())
    return X,df

def evaluate(X,df,out):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix
    reports=[]; predictions=[]
    # Evaluate pooled model and MV7-only model separately; each test fold has unseen groups.
    for scope in ('pooled','mv7'):
        ids=np.flatnonzero(np.ones(len(df),bool) if scope=='pooled' else df.dataset.eq('mv7').to_numpy())
        sub=df.iloc[ids]; y=sub.label.to_numpy(); g=sub.group.to_numpy()
        n=min(5,int(sub.groupby('label').group.nunique().min()))
        if n<2 or set(y)!=set(LABELS): raise ValueError(f'Insufficient classes/groups for {scope}')
        splits=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=42).split(X[ids],y,g))
        for name,model in [('LR',make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',C=1))),('RF',RandomForestClassifier(n_estimators=200,min_samples_leaf=3,class_weight='balanced_subsample',n_jobs=-1,random_state=42))]:
            for fold,(tr,te) in enumerate(splits):
                assert not set(g[tr])&set(g[te]); assert set(y[tr])==set(LABELS),f'Missing training class: {scope} {fold}'
                model.fit(X[ids[tr]],y[tr]); pred=model.predict(X[ids[te]])
                z=sub.iloc[te].copy(); z['prediction']=pred; z['scope']=scope; z['model']=name; z['fold']=fold; predictions.append(z)
            z=pd.concat([p for p in predictions if p.scope.iloc[0]==scope and p.model.iloc[0]==name]);
            for domain in ['all']+sorted(z.dataset.unique()):
                a=z if domain=='all' else z[z.dataset==domain]; labels=[l for l in LABELS if l in set(a.label)]
                reports.append(dict(scope=scope,model=name,domain=domain,windows=len(a),macro_f1=f1_score(a.label,a.prediction,labels=labels,average='macro',zero_division=0),balanced_accuracy=balanced_accuracy_score(a.label,a.prediction),confusion=confusion_matrix(a.label,a.prediction,labels=LABELS).tolist(),labels=LABELS))
    pd.concat(predictions).to_csv(out/'oof_predictions.csv',index=False); (out/'metrics.json').write_text(json.dumps(reports,indent=2)); print(pd.DataFrame(reports).drop(columns=['confusion','labels']).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seconds',type=int,choices=[2,4],default=2); p.add_argument('--build-only',action='store_true'); a=p.parse_args()
    out=Path('results/exercise_baseline')/f'{a.seconds}s'; out.mkdir(parents=True,exist_ok=True)
    X,df=build(a.seconds,out)
    if not a.build_only: evaluate(X,df,out)
    print('Saved:',out)
