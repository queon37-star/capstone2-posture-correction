#!/usr/bin/env python3
"""Fixed full-interval LOPO comparison: LR180 vs relative geometry vs combined.
Reads existing YOLO caches only. No pose inference or deployment changes.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from train_squat_error_baseline import CODES, sha, metrics


def stat(a, q=.5):
    a=np.asarray(a); a=a[np.isfinite(a)]
    return float(np.quantile(a,q)) if len(a) else np.nan


def midpoint(a,b):
    return (a+b)/2


def angle(a,b,c):
    u=a-b; v=c-b
    den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.where(den>1e-6,np.degrees(np.arccos(np.clip(np.sum(u*v,axis=1)/den,-1,1))),np.nan)


def geometry(seq):
    k=np.asarray(seq,float); xy=k[:,:,:2].copy()
    xy[~(np.isfinite(k).all(2)&(k[:,:,2]>=.3))]=np.nan
    S,H,K,A=[midpoint(xy[:,l],xy[:,r]) for l,r in [(5,6),(11,12),(13,14),(15,16)]]
    n=len(k); m=max(3,int(np.ceil(.15*n))); start=slice(0,m)
    leg=np.concatenate([np.linalg.norm(xy[:,h]-xy[:,j],axis=1)+np.linalg.norm(xy[:,j]-xy[:,a],axis=1) for h,j,a in [(11,13,15),(12,14,16)]])
    scale=stat(leg)
    if not np.isfinite(scale) or scale<=10: raise ValueError('No usable leg scale')
    hip=H[:,1].copy(); ok=np.isfinite(hip)
    if ok.sum()<3: raise ValueError('No usable hip trajectory')
    v=np.interp(np.arange(n),np.flatnonzero(ok),hip[ok]); sm=np.array([np.median(v[max(0,i-3):i+4]) for i in range(n)])
    b=int(np.argmax(sm)); radius=max(2,int(round(.05*n))); bottom=slice(max(0,b-radius),min(n,b+radius+1))
    def sep(l,r): return np.abs(xy[:,l,0]-xy[:,r,0])/scale
    signals={'hip_height_rel_ankle':(A[:,1]-H[:,1])/scale,
             'torso_x_abs':np.abs(S[:,0]-H[:,0])/scale,
             'torso_x_signed':(S[:,0]-H[:,0])/scale,
             'shoulder_level_abs':np.abs(xy[:,5,1]-xy[:,6,1])/scale,
             'hip_level_abs':np.abs(xy[:,11,1]-xy[:,12,1])/scale,
             'knee_width':sep(13,14),'ankle_width':sep(15,16),
             'knee_minus_ankle_width':sep(13,14)-sep(15,16),
             'knee_center_offset':np.abs(K[:,0]-A[:,0])/scale}
    width=sep(15,16)
    signals['knee_ankle_ratio']=np.divide(sep(13,14),width,out=np.full(n,np.nan),where=width>=.15)
    # Absolute projected offsets are proxies, not foot/toe or heel measurements.
    for side,(h,j,a) in [('l',(11,13,15)),('r',(12,14,16))]:
        signals['knee_angle_'+side]=angle(xy[:,h],xy[:,j],xy[:,a])/180
        signals['knee_ankle_x_'+side]=np.abs(xy[:,j,0]-xy[:,a,0])/scale
        signals['ankle_y_'+side]=xy[:,a,1]/scale
        signals['hip_knee_y_'+side]=(xy[:,j,1]-xy[:,h,1])/scale
        # inward movement relative to ankle midpoint, masked for overlapping ankles.
        inward=(np.abs(xy[:,a,0]-A[:,0])-np.abs(xy[:,j,0]-A[:,0]))/scale
        inward[~(width>=.15)]=np.nan
        signals['inward_'+side]=inward
    # Symmetric aggregates avoid reliance on left vs right anatomy; use both
    # signed change and absolute change from the start of the repetition.
    features={}
    for key,x in signals.items():
        ref=stat(x[start]); delta=x-ref
        for suffix,val in [('delta_p10',stat(delta,.1)),('delta_p90',stat(delta,.9)),('abs_delta_p90',stat(np.abs(delta),.9)),('bottom_delta',stat(delta[bottom])),('range',stat(x,.9)-stat(x,.1))]:
            features[key+'_'+suffix]=val
        # No absolute pixel y location: ankle_y only relative temporal features.
        if not key.startswith('ankle_y_'):
            features[key+'_bottom']=stat(x[bottom])
    return features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,default=Path('results/squat_multiview/build'))
    p.add_argument('--out-dir',type=Path,default=Path('results/squat_error_geometry'))
    a=p.parse_args(); b=a.build_dir
    m=json.loads((b/'manifest.json').read_text())
    for name,key in [('features_crop10.npy','features_sha256'),('repetitions.csv','repetitions_sha256')]:
        if sha(b/name)!=m[key]: raise ValueError('Hash mismatch: '+name)
    ep=Path('data/preprocess/extra_datasets/manifest.json')
    if sha(ep)!=m['rehab26_manifest_sha256']: raise ValueError('Cache manifest differs from feature build')
    em=json.loads(ep.read_text()); assert em['weights_sha256']==m['pose_weights_sha256']
    x=np.load(b/'features_crop10.npy',allow_pickle=False); d=pd.read_csv(b/'repetitions.csv'); assert x.shape==(len(d),4,180)
    d['row']=np.arange(len(d)); d=d[(d.dataset=='rehab26')&(d.label==0)].copy().reset_index(drop=True)
    assert len(d)==834 and d.rep_id.nunique()==278 and d.person_id.nunique()==5 and not d.duplicated(['rep_id','camera']).any()
    assert (d.groupby('rep_id').camera.nunique()==3).all()
    d['truth']=d.error_description; assert set(d.truth)==set(CODES)
    d['view_bin']=pd.cut(d.viewing_angle_deg.abs(),[-1,30,60,90],labels=['frontal_0_30','oblique_gt30_60','profile_gt60_90']); assert d.view_bin.notna().all()
    import preprocess_rehab24 as pre
    from extract_video_kp_cache import KP_DUMP_DTYPE
    from build_squat_multiview import assess
    vids={v['sample_id']:v for v in em['videos'] if v['dataset']=='rehab26' and int(v['exercise_id'])==6}; assert len(vids)==15
    records={}
    for sid,rows in d.groupby('sample_id'):
        v=vids[sid]; assert pre.check_cache(v,m['pose_weights_sha256'])=='complete'; assert abs(v['fps']-30)<.01
        paths=pre.cache_paths(v); k=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r'); assert np.array_equal(k['frame_id'],np.arange(len(k)))
        with np.load(paths[2],allow_pickle=False) as z: status=z['status']; boxes=z['selected_box']
        for idx,r in rows.iterrows():
            lo,hi=int(r.first_frame),int(r.last_frame); assert 0<=lo<=hi<len(k)
            seq,_,why=assess(k['kp'],status,boxes,lo,hi)
            if why: raise ValueError(f'Previously accepted row now rejected: {sid}, {lo}, {why}')
            records[idx]=geometry(seq)
        print('Features',sid,len(rows),flush=True)
    g=pd.DataFrame.from_dict(records,orient='index').sort_index(); assert g.index.tolist()==list(range(len(d))) and not np.isinf(g.to_numpy()).any()
    base=x[d.row.to_numpy(),0,:]; new=g.to_numpy(); data={'baseline180':base,'relative_geometry':new,'combined':np.concatenate([base,new],axis=1)}
    a.out_dir.mkdir(parents=True,exist_ok=False)
    g.to_csv(a.out_dir/'geometry_features.csv',index=False)
    preds=[]; folds=[]; classes=[]; views=[]; conf=[]; abst=[]; summary={}
    for mode,xx in data.items():
        parts=[]
        for person in sorted(d.person_id.unique()):
            tr=d.person_id.ne(person).to_numpy(); te=~tr
            assert set(d[tr].rep_id).isdisjoint(d[te].rep_id)
            mdl=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight='balanced',random_state=42))
            mdl.fit(xx[tr],d.loc[tr,'truth']); z=d[te].copy(); pr=mdl.predict_proba(xx[te]); z['prediction']=mdl.classes_[pr.argmax(1)];z['max_probability']=pr.max(1);z['mode']=mode
            parts.append(z); f=dict(mode=mode,heldout_person=person,**metrics(z));folds.append(f);print(mode,person,f['accuracy'],flush=True)
        pred=pd.concat(parts,ignore_index=True);preds.append(pred); summary[mode]=dict(n_features=xx.shape[1],**metrics(pred))
        pr,rc,f1,sup=precision_recall_fscore_support(pred.truth,pred.prediction,labels=CODES,zero_division=0)
        for i,c in enumerate(CODES):classes.append(dict(mode=mode,error=c,precision=pr[i],recall=rc[i],f1=f1[i],n_views=int(sup[i])))
        for view,z in pred.groupby('view_bin',observed=True):
            views.append(dict(mode=mode,view=str(view),error='ALL',**metrics(z)))
            for c in CODES:
                q=z[z.truth.eq(c)];views.append(dict(mode=mode,view=str(view),error=c,n_views=len(q),accuracy=float(q.truth.eq(q.prediction).mean())))
        cm=confusion_matrix(pred.truth,pred.prediction,labels=CODES)
        for i,t in enumerate(CODES):
            for j,h in enumerate(CODES): conf.append(dict(mode=mode,truth=t,prediction=h,count=int(cm[i,j])))
        for threshold in [0,.5,.7,.9]:
            keep=pred.max_probability.ge(threshold); correct=pred.truth.eq(pred.prediction)
            abst.append(dict(mode=mode,threshold=threshold,coverage=float(keep.mean()),reported_accuracy=float(correct[keep].mean()) if keep.any() else None))
    for name,records in [('folds',folds),('per_class',classes),('by_view',views),('confusion',conf),('abstention',abst)]:pd.DataFrame(records).to_csv(a.out_dir/(name+'.csv'),index=False)
    pd.concat(preds,ignore_index=True).to_csv(a.out_dir/'predictions.csv',index=False)
    (a.out_dir/'summary.json').write_text(json.dumps(dict(scope='GT bad, full intervals, person-disjoint LOPO; exploratory comparison on same five people',feature_names=list(g.columns),source_hashes={k:m[k] for k in ['features_sha256','repetitions_sha256','rehab26_manifest_sha256']},results=summary),indent=2))
    print(json.dumps(summary,indent=2));print('Saved:',a.out_dir)

if __name__=='__main__':main()
