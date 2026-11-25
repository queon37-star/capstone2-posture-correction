#!/usr/bin/env python3
"""Two predeclared source/class-weighted LR candidates, frozen GT LOPO protocol.
Keep feature schema, Crop10, imputer/scaler (unweighted), LR C and default threshold.
Weights affect only LR loss; each dataset gets specified mass, split equally by label.
Existing A/B/C/D predictions are reference results, not re-trained.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from train_squat_multiview import digest, validate, report

CANDIDATES = {"D_w50": .50, "D_w75": .75}

def loss_weights(d, rehab24_share):
    """Per-view weights, each repeated four times for Crop10; mean = 1."""
    weights = np.zeros(len(d), dtype=np.float64)
    for ds, mass in [('rehab24', rehab24_share), ('rehab26', 1-rehab24_share)]:
        for lab in (0, 1):
            sel = (d.dataset.eq(ds) & d.label.eq(lab)).to_numpy()
            if not sel.any():
                raise ValueError(f"Training fold missing {ds} label {lab}")
            weights[sel] = len(d)*mass/2/sel.sum()
    if not np.isclose(weights.mean(), 1) or (weights <= 0).any():
        raise ValueError("Invalid weights")
    return weights

def collect(z):
    metrics, subjects, angles, errors = [], [], [], []
    for v,t in z.groupby('variant',sort=False):
        for domain, s in [('all',t), ('rehab24',t[t.dataset=='rehab24']),
                          ('rehab24_cam17',t[(t.dataset=='rehab24')&(t.camera==17)]),
                          ('rehab24_cam18',t[(t.dataset=='rehab24')&(t.camera==18)]),
                          ('rehab26',t[t.dataset=='rehab26'])]:
            metrics.append(dict(variant=v,domain=domain,**report(s)))
        for g,s in t.groupby('group'):
            subjects.append(dict(variant=v,group=g,**report(s)))
        r=t[t.dataset=='rehab26'].copy()
        r['angle_bin']=pd.cut(r.viewing_angle_deg.abs(),[-.01,15,60,90],
                              labels=['0-15','>15-60','>60-90'])
        if r.angle_bin.isna().any(): raise ValueError('Invalid viewing angle')
        for a,s in r.groupby('angle_bin',observed=True):
            angles.append(dict(variant=v,angle_bin=str(a),**report(s)))
        for e,s in r[r.label==0].groupby('error_description',dropna=False):
            errors.append(dict(variant=v,error_description=str(e),n=len(s),
                               bad_correct=int(s.pred.eq(0).sum()),bad_recall=float(s.pred.eq(0).mean())))
    return metrics,subjects,angles,errors

def run(args):
    b,o,c=args.build_dir,args.out_dir,args.compare_dir
    if o.exists(): raise FileExistsError(o)
    m=json.loads((b/'manifest.json').read_text())
    for path,key in [(b/'features_crop10.npy','features_sha256'),
                     (b/'repetitions.csv','repetitions_sha256'),
                     (args.template,'template_sha256')]:
        if digest(path)!=m[key]: raise ValueError(f'Hash mismatch: {path}')
    d=pd.read_csv(b/'repetitions.csv',dtype={'person_id':str,'group':str,'rep_id':str})
    x=np.load(b/'features_crop10.npy',allow_pickle=False)
    validate(d,x,m)
    old_summary=json.loads((c/'summary.json').read_text())
    if old_summary['feature_sha256']!=m['features_sha256']: raise ValueError('Baseline feature mismatch')
    checks=old_summary['reference_checks']
    if len(checks)!=9 or not all(a['same_predictions'] and a['max_abs_prob_diff']<=1e-6 for a in checks):
        raise ValueError('Baseline A reproduction did not pass')
    old=pd.read_csv(c/'predictions.csv',dtype={'person_id':str,'group':str,'rep_id':str})
    keys=['rep_id','camera','label','group','dataset','sample_id','first_frame','last_frame']
    expected=d[keys].sort_values(['rep_id','camera']).reset_index(drop=True)
    if set(old.variant)!={'A','B','C','D'}: raise ValueError('Wrong baseline variants')
    for v,t in old.groupby('variant'):
        pd.testing.assert_frame_equal(t[keys].sort_values(['rep_id','camera']).reset_index(drop=True),expected)
    if not old.pred.isin([0,1]).all(): raise ValueError('Invalid reference predictions')
    source=joblib.load(args.template) # trusted self-trained artifact only
    names=source['feature_names']; template=source['pipeline']
    if source.get('feature_variant')!='PlusAsymmetry180' or list(names)!=m['feature_names'] or len(set(names))!=180:
        raise ValueError('Feature schema mismatch')
    if template.get_params().get('model__class_weight')!='balanced':
        raise ValueError('Unexpected baseline class weight')
    y=d.label.to_numpy(int); groups=d.group.to_numpy()
    o.mkdir(parents=True,exist_ok=False)
    preds=[old]; folds=[]
    for variant,share in CANDIDATES.items():
        for g in sorted(set(groups)):
            tr=groups!=g; te=~tr
            train=d.loc[tr]
            if set(train.group)&set(d.loc[te,'group']) or set(train.rep_id)&set(d.loc[te,'rep_id']):
                raise RuntimeError('Train/test leakage')
            w=loss_weights(train,share)
            xx=x[tr].reshape(-1,180); yy=np.repeat(y[tr],4); ww=np.repeat(w,4)
            # Avoid implicit double class weighting; explicit weights are balanced
            # within each source and total mean weight stays one.
            model=clone(template).set_params(model__class_weight=None)
            model.fit(xx,yy,model__sample_weight=ww)
            xt=x[te,0,:]
            row=d.loc[te].copy(); row.insert(0,'variant',variant)
            row['pred']=model.predict(xt).astype(int)
            row['prob_correct']=model.predict_proba(xt)[:,list(model.classes_).index(1)]
            row['train_view_reps']=int(tr.sum()); row['evaluation_type']='person_held_out'
            preds.append(row)
            cell={}
            for ds in ('rehab24','rehab26'):
                for lab in (0,1):
                    mask=(train.dataset.eq(ds)&train.label.eq(lab)).to_numpy()
                    cell[f'{ds}_label{lab}_loss_share']=float(w[mask].sum()/w.sum())
            f=dict(variant=variant,held_out_group=g,train_view_reps=int(tr.sum()),
                   train_unique_reps=int(train.rep_id.nunique()),train_feature_rows=len(yy),
                   weight_min=float(w.min()),weight_max=float(w.max()),**cell,**report(row))
            folds.append(f)
            if g.startswith('rehab24:'):
                cam17=te&d.dataset.eq('rehab24').to_numpy()&d.camera.eq(17).to_numpy()
                vids=d.loc[cam17,'video_id'].unique()
                if len(vids)!=1: raise ValueError('Missing Cam17 video')
                vid=str(vids[0]); person=int(g.split(':')[1])
                art=dict(pipeline=model,feature_names=list(names),feature_groups=['base','asymmetry'],
                         feature_variant='PlusAsymmetry180',training_variant=variant+'_crop10_lopo',
                         crop_fraction=.10,label_map={0:'incorrect',1:'correct'},
                         held_out_video=vid,held_out_person_id=person,held_out_group=g,
                         training_person_ids=sorted(int(s.split(':')[1]) for s in set(groups[tr]) if s.startswith('rehab24:')),
                         training_groups=sorted(set(groups[tr])),training_repetitions=int(tr.sum()),
                         training_unique_repetitions=int(train.rep_id.nunique()),training_feature_rows=len(yy),
                         features_sha256=m['features_sha256'],rehab24_loss_share=share,
                         evaluation_note='Exploratory source/class weighted loss; unweighted preprocessing; held-out GT.')
                path=o/'models'/variant/f'{vid}.joblib'; path.parent.mkdir(parents=True,exist_ok=True)
                joblib.dump(art,path)
                if not np.array_equal(joblib.load(path)['pipeline'].predict_proba(xt),model.predict_proba(xt)):
                    raise RuntimeError('Reload mismatch')
            print(f'{variant} {g}: train={tr.sum()} test={te.sum()} acc={f["accuracy"]:.4f}',flush=True)
    z=pd.concat(preds,ignore_index=True)
    if len(z)!=6*len(d) or z.duplicated(['variant','rep_id','camera']).any() or len(folds)!=28:
        raise RuntimeError('Prediction coverage mismatch')
    metrics,subjects,angles,errors=collect(z)
    table=pd.DataFrame(metrics)
    domain_mean=table[table.domain.isin(['rehab24','rehab26'])].groupby('variant',sort=False).agg(
        equal_dataset_balanced_accuracy=('balanced_accuracy','mean'),
        equal_dataset_bad_recall=('bad_recall','mean')).reset_index()
    for name,rows in [('metrics',metrics),('per_person',subjects),('per_angle',angles),('per_error',errors),('folds',folds)]:
        pd.DataFrame(rows).to_csv(o/f'{name}.csv',index=False)
    z.to_csv(o/'predictions.csv',index=False)
    domain_mean.to_csv(o/'equal_dataset_metrics.csv',index=False)
    (o/'summary.json').write_text(json.dumps(dict(feature_sha256=m['features_sha256'],
        baseline_predictions_sha256=digest(c/'predictions.csv'),candidate_rehab24_loss_shares=CANDIDATES,
        metrics=metrics,equal_dataset_metrics=domain_mean.to_dict('records'),
        notes='Exploratory same held-out subjects. Weights affect LR loss only; imputer/scaler unchanged, each source class gets half its source mass. Mean weight one, C and threshold unchanged. All views and crops remain within training fold. A/B/C/D copied from original predictions. Not independent final test.'),indent=2))
    print(table[table.domain.isin(['rehab24_cam17','rehab24_cam18','rehab24','rehab26'])][['variant','domain','accuracy','balanced_accuracy','bad_recall','bad_precision']].to_string(index=False))
    print(domain_mean.to_string(index=False)); print('Saved:',o)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,default=Path('results/squat_multiview/build'))
    p.add_argument('--compare-dir',type=Path,default=Path('results/squat_multiview/compare'))
    p.add_argument('--template',type=Path,default=Path('models/squat_plus_asymmetry180.joblib'))
    p.add_argument('--out-dir',type=Path,default=Path('results/squat_multiview/weighted'))
    run(p.parse_args())

if __name__=='__main__': main()
