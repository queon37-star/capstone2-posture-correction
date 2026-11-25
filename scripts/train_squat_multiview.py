#!/usr/bin/env python3
"""Compare A/B/C/D with fixed 180-feature Crop10 LR; held-out full GT only.
Run in pose_training_offline. Inputs from build_squat_multiview.py.
All views of a held-out participant excluded, including for Camera17 evaluation.
A/B -> REHAB26 is cross-dataset transfer; C/D -> REHAB26 is participant LOPO.
Only REHAB24 fold models saved, for later unchanged streaming evaluator.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score


def digest(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def report(d):
    y=d.label.to_numpy(int); p=d.pred.to_numpy(int)
    tn=int(((y==1)&(p==1)).sum()); fp=int(((y==1)&(p==0)).sum())
    tp=int(((y==0)&(p==0)).sum()); fn=int(((y==0)&(p==1)).sum())
    def div(a,b): return float(a/b) if b else None
    br=div(tp,tp+fn); gr=div(tn,tn+fp)
    return dict(n=len(y),unique_reps=int(d.rep_id.nunique()),groups=int(d.group.nunique()),good=tn+fp,bad=tp+fn,correct=tn+tp,accuracy=div(tn+tp,len(y)),bad_recall=br,bad_precision=div(tp,tp+fp),good_recall=gr,balanced_accuracy=(br+gr)/2 if br is not None and gr is not None else None,macro_f1=float(f1_score(y,p,labels=[0,1],average='macro',zero_division=0)) if br is not None and gr is not None else None,good_as_good=tn,good_as_bad=fp,bad_as_good=fn,bad_as_bad=tp)


def masks(d):
    r17=(d.dataset.eq('rehab24') & d.camera.eq(17)).to_numpy()
    r18=(d.dataset.eq('rehab24') & d.camera.eq(18)).to_numpy()
    r26=d.dataset.eq('rehab26').to_numpy()
    return {'A':r17,'B':r17|r18,'C':r17|r26,'D':r17|r18|r26}


def validate(d,x,m):
    if list(x.shape)!=m['feature_shape'] or x.shape!=(len(d),4,180) or np.isinf(x).any() or not np.isfinite(x).any(axis=2).all(): raise ValueError('Invalid feature array')
    if m['augmentation']!=['full','trim_start','trim_end','trim_both'] or m['label_map']!={'0':'bad','1':'good'}: raise ValueError('Wrong augmentation/label schema')
    keys=['dataset','camera','person_id','group','video_id','rep_id','label','repetition_number','first_frame','last_frame']
    if d[keys].isna().any().any() or d.duplicated(['rep_id','camera']).any(): raise ValueError('Missing/duplicate repetition metadata')
    if not d.group.eq(d.dataset+':'+d.person_id).all(): raise ValueError('Group mismatch')
    if (d.groupby('rep_id')[['label','group','first_frame','last_frame']].nunique()>1).any().any(): raise ValueError('Cross-view metadata mismatch')
    if set(d.label)!={0,1} or len(d)!=2048 or d.rep_id.nunique()!=749 or d.group.nunique()!=14: raise ValueError('Unexpected corpus inventory')
    expected={('rehab24',17,0):61,('rehab24',17,1):134,('rehab24',18,0):58,('rehab24',18,1):133,**{('rehab26',c,l):n for c in (1,2,3) for l,n in [(0,278),(1,276)]}}
    if d.groupby(['dataset','camera','label']).size().to_dict()!=expected: raise ValueError('Class counts changed')
    if d[d.dataset.eq('rehab24') & d.camera.eq(17)].group.nunique()!=9 or d[d.dataset.eq('rehab26')].group.nunique()!=5: raise ValueError('Person counts changed')


def run(args):
    b=args.build_dir; o=args.out_dir
    if o.exists(): raise FileExistsError(o)
    m=json.loads((b/'manifest.json').read_text())
    for p,key in [(b/'features_crop10.npy','features_sha256'),(b/'repetitions.csv','repetitions_sha256'),(args.template,'template_sha256')]:
        if digest(p)!=m[key]: raise ValueError(f'Hash mismatch: {p}')
    d=pd.read_csv(b/'repetitions.csv',dtype={'person_id':str,'group':str,'rep_id':str}); x=np.load(b/'features_crop10.npy',allow_pickle=False)
    validate(d,x,m)
    source=joblib.load(args.template) # only trusted self-trained artifacts
    names=source['feature_names']
    if source.get('feature_variant')!='PlusAsymmetry180' or list(names)!=m['feature_names'] or len(set(names))!=180: raise ValueError('Feature schema mismatch')
    template=source['pipeline']; y=d.label.to_numpy(int); group=d.group.to_numpy(); r24=d.dataset.eq('rehab24').to_numpy()
    # Reproduce A against the original saved nine models before other variants.
    o.mkdir(parents=True,exist_ok=False)
    preds=[]; folds=[]; checks=[]
    for variant,pool in masks(d).items():
        for g in sorted(set(group)):
            tr=pool & (group!=g); te=group==g
            if set(group[tr]) & set(group[te]) or set(d.loc[tr,'rep_id']) & set(d.loc[te,'rep_id']): raise RuntimeError('Train/test leakage')
            xx=x[tr].reshape(-1,180); yy=np.repeat(y[tr],4)
            if set(yy)!={0,1}: raise ValueError('Training missing class')
            model=clone(template).fit(xx,yy)
            xt=x[te,0,:]; pred=model.predict(xt).astype(int); prob=model.predict_proba(xt)[:,list(model.classes_).index(1)]
            row=d.loc[te].copy(); row.insert(0,'variant',variant); row['pred']=pred; row['prob_correct']=prob; row['train_view_reps']=int(tr.sum()); row['evaluation_type']='cross_dataset' if g.startswith('rehab26:') and variant in ('A','B') else 'person_held_out'
            preds.append(row)
            f=dict(variant=variant,held_out_group=g,train_view_reps=int(tr.sum()),train_unique_reps=int(d.loc[tr,'rep_id'].nunique()),train_groups=int(d.loc[tr,'group'].nunique()),train_feature_rows=len(yy),train_good=int(y[tr].sum()),train_bad=int((y[tr]==0).sum()),**report(row))
            folds.append(f)
            if g.startswith('rehab24:'):
                cam17=te&r24&d.camera.eq(17).to_numpy(); vid=d.loc[cam17,'video_id'].unique()
                if len(vid)!=1: raise ValueError('Expected single Camera17 video per group')
                vid=str(vid[0]); person=int(g.split(':')[1]); p=o/'models'/variant/f'{vid}.joblib'; p.parent.mkdir(parents=True,exist_ok=True)
                if variant=='A':
                    old=joblib.load(args.reference_dir/f'{vid}.joblib')
                    if old.get('held_out_video')!=vid or int(old.get('held_out_person_id',-1))!=person or list(old['feature_names'])!=list(names): raise ValueError(f'{vid}: old reference metadata mismatch')
                    oldp=old['pipeline']; checkx=x[cam17,0,:]
                    diff=float(np.max(np.abs(oldp.predict_proba(checkx)-model.predict_proba(checkx))))
                    same=bool(np.array_equal(oldp.predict(checkx),model.predict(checkx)))
                    checks.append(dict(video_id=vid,max_abs_prob_diff=diff,same_predictions=same))
                    if not same or diff>1e-6: raise RuntimeError(f'A failed reference reproduction for {vid}: {diff}, {same}')
                art=dict(pipeline=model,feature_names=list(names),feature_groups=['base','asymmetry'],feature_variant='PlusAsymmetry180',training_variant=f'multiview_{variant}_crop10_lopo',crop_fraction=.10,label_map={0:'incorrect',1:'correct'},held_out_video=vid,held_out_person_id=person,held_out_group=g,training_person_ids=sorted(int(s.split(':')[1]) for s in set(group[tr]) if s.startswith('rehab24:')),training_groups=sorted(set(group[tr])),training_repetitions=int(tr.sum()),training_unique_repetitions=int(d.loc[tr,'rep_id'].nunique()),training_feature_rows=len(yy),features_sha256=m['features_sha256'],evaluation_note='Exploratory GT-segment LOPO; not independent final test.')
                joblib.dump(art,p); reload=joblib.load(p)['pipeline'];
                if not np.array_equal(reload.predict_proba(xt),model.predict_proba(xt)): raise RuntimeError('Reload mismatch')
            print(f'{variant} {g}: train={int(tr.sum())} views / {len(yy)} rows, test={int(te.sum())}, acc={f["accuracy"]:.4f}',flush=True)
    z=pd.concat(preds,ignore_index=True)
    if len(z)!=4*len(d) or z.duplicated(['variant','rep_id','camera']).any() or len(checks)!=9: raise RuntimeError('Prediction coverage mismatch')
    results=[]; errors=[]; angles=[]; subjects=[]
    for v,t in z.groupby('variant',sort=True):
        for name,s in [('all',t),('rehab24',t[t.dataset=='rehab24']),('rehab24_cam17',t[(t.dataset=='rehab24')&(t.camera==17)]),('rehab24_cam18',t[(t.dataset=='rehab24')&(t.camera==18)]),('rehab26',t[t.dataset=='rehab26'])]:
            results.append(dict(variant=v,domain=name,**report(s)))
        for (ds,c),s in t.groupby(['dataset','camera']):
            if ds=='rehab26': results.append(dict(variant=v,domain=f'rehab26_cam{c}',**report(s)))
        for g,s in t.groupby('group'): subjects.append(dict(variant=v,group=g,**report(s)))
        r=t[t.dataset=='rehab26'].copy(); angle=r.viewing_angle_deg.abs(); r['angle_bin']=pd.cut(angle,[-.01,15,60,90],labels=['0-15','>15-60','>60-90'])
        if r.angle_bin.isna().any(): raise ValueError('Invalid REHAB26 viewing angle')
        for a,s in r.groupby('angle_bin',observed=True): angles.append(dict(variant=v,angle_bin=str(a),**report(s)))
        for e,s in r[r.label==0].groupby('error_description',dropna=False): errors.append(dict(variant=v,error_description=str(e),n=len(s),bad_correct=int(s.pred.eq(0).sum()),bad_recall=float(s.pred.eq(0).mean())))
    pd.DataFrame(results).to_csv(o/'metrics.csv',index=False); pd.DataFrame(subjects).to_csv(o/'per_person.csv',index=False); pd.DataFrame(angles).to_csv(o/'per_angle.csv',index=False); pd.DataFrame(errors).to_csv(o/'per_error.csv',index=False); pd.DataFrame(folds).to_csv(o/'folds.csv',index=False); z.to_csv(o/'predictions.csv',index=False)
    (o/'summary.json').write_text(json.dumps(dict(reference_checks=checks,metrics=results,feature_sha256=m['features_sha256'],notes='All four evaluated on identical GT views. A/B on REHAB26 are cross-dataset transfer, C/D are mixed-data LOPO. No test augmentation; equal weight per training view and crop; 36 REHAB24 held-out models saved; no final deployment model. Labels are dataset-specific.'),indent=2))
    print('\nA reference reproduction: PASS (9/9)')
    print(pd.DataFrame(results).query('domain in ["rehab24_cam17","rehab24_cam18","rehab26"]')[['variant','domain','n','accuracy','balanced_accuracy','bad_recall','bad_precision']].to_string(index=False)); print('Saved:',o)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--build-dir',type=Path,default=Path('results/squat_multiview/build')); p.add_argument('--template',type=Path,default=Path('models/squat_plus_asymmetry180.joblib')); p.add_argument('--reference-dir',type=Path,default=Path('models/lopo_heldout/PlusAsymmetry180_crop10')); p.add_argument('--out-dir',type=Path,default=Path('results/squat_multiview/compare')); run(p.parse_args())

if __name__=='__main__': main()
