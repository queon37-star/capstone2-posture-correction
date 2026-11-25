#!/usr/bin/env python3
"""Conditional 5-class squat error baseline, with person-disjoint LOPO.
Reuse the existing full/Crop10 features. No change to deployment models.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CODES = ['ERR_KNEES_INSIDE', 'ERR_TORSO_TO_SIDE', 'ERR_SHALLOW_SQUAT',
         'ERR_KNEES_FORWARD', 'ERR_LIFTED_HEELS']


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def metrics(d):
    if not len(d):
        return dict(n_views=0, n_reps=0, accuracy=None, macro_f1=None, balanced_accuracy=None)
    return dict(n_views=len(d), n_reps=d.rep_id.nunique(),
                accuracy=float(accuracy_score(d.truth, d.prediction)),
                macro_f1=float(f1_score(d.truth, d.prediction, labels=CODES, average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(d.truth, d.prediction)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, default=Path('results/squat_multiview/build'))
    p.add_argument('--annotation', type=Path, default=Path.home()/'capstone2/datasets/REHAB26-ViewAngles/Segmentation.csv')
    p.add_argument('--out-dir', type=Path, default=Path('results/squat_error_baseline'))
    args=p.parse_args()
    b=args.build_dir
    m=json.loads((b/'manifest.json').read_text())
    for f,key in [('features_crop10.npy','features_sha256'),('repetitions.csv','repetitions_sha256')]:
        if sha(b/f)!=m[key]: raise ValueError(f'Hash mismatch: {f}')
    x=np.load(b/'features_crop10.npy',allow_pickle=False)
    d=pd.read_csv(b/'repetitions.csv')
    assert list(x.shape)==m['feature_shape']==[len(d),4,180]
    assert m['augmentation']==['full','trim_start','trim_end','trim_both']
    assert len(set(m['feature_names']))==180
    assert not np.isinf(x).any()
    d['row']=np.arange(len(d))
    r=d[d.dataset.eq('rehab26')].copy()
    assert len(r)==1662 and r.rep_id.nunique()==554 and r.person_id.nunique()==5
    assert not r.duplicated(['rep_id','camera']).any()
    assert (r.groupby('rep_id').camera.nunique()==3).all()
    assert (r.groupby('rep_id').person_id.nunique()==1).all()
    # Verify every cached view against original annotation, not row position.
    a=pd.read_csv(args.annotation,sep=';')
    a=a[a.exercise_id.eq(6)].copy()
    assert len(a)==554 and set(a.correctness_score)=={1,5}
    assert a.additional_errors.fillna('').str.strip().eq('').all()
    a['rep_id']='rehab26:'+a.video_id.astype(str)+':'+a.person_orientation.astype(int).astype(str)+':'+a.repetition_number.astype(int).astype(str)
    assert not a.rep_id.duplicated().any() and set(a.rep_id)==set(r.rep_id)
    a=a.set_index('rep_id')
    for cached,raw in [('person_id','person'),('first_frame','first_frame'),('last_frame','last_frame'),('orientation','person_orientation'),('error_description','error_description')]:
        assert np.array_equal(r[cached].fillna('').to_numpy(),r.rep_id.map(a[raw]).fillna('').to_numpy()), f'Mismatch: {cached}'
    assert np.array_equal(r.label.to_numpy(), r.rep_id.map(a.correctness_score).eq(1).astype(int).to_numpy())
    angle=(r.camera.astype(int)-1)*45-r.orientation
    assert np.allclose(r.viewing_angle_deg,angle)
    bad=r[r.label.eq(0)].copy().reset_index(drop=True)
    assert len(bad)==834 and bad.rep_id.nunique()==278 and set(bad.error_description)==set(CODES)
    bad['truth']=bad.error_description
    bad['view_bin']=pd.cut(bad.viewing_angle_deg.abs(),[-1,30,60,90],labels=['frontal_0_30','oblique_gt30_60','profile_gt60_90'])
    assert bad.view_bin.notna().all()
    args.out_dir.mkdir(parents=True,exist_ok=False)
    all_preds=[]; fold_rows=[]; classes=[]; conf=[]; views=[]; coverage=[]; summary={}
    for mode in ['full','crop10']:
        parts=[]
        for person in sorted(bad.person_id.unique()):
            tr=bad[bad.person_id.ne(person)]; te=bad[bad.person_id.eq(person)].copy()
            assert set(tr.rep_id).isdisjoint(te.rep_id) and set(tr.truth)==set(te.truth)==set(CODES)
            xx=x[tr.row.to_numpy(),:1 if mode=='full' else 4,:]
            yy=np.repeat(tr.truth.to_numpy(),xx.shape[1])
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight='balanced',random_state=42))
            model.fit(xx.reshape(-1,180),yy)
            prob=model.predict_proba(x[te.row.to_numpy(),0,:])
            for i,c in enumerate(model.classes_): te['prob_'+c]=prob[:,i]
            te['prediction']=model.classes_[prob.argmax(axis=1)]
            te['max_probability']=prob.max(axis=1)
            te['mode']=mode
            parts.append(te)
            row=dict(mode=mode,heldout_person=person,train_views=len(tr),train_rows=len(yy),**metrics(te))
            fold_rows.append(row)
            print(mode,person,'accuracy',round(row['accuracy'],4),'macro F1',round(row['macro_f1'],4),flush=True)
        pred=pd.concat(parts,ignore_index=True)
        assert len(pred)==len(bad) and not pred.duplicated(['rep_id','camera']).any()
        all_preds.append(pred)
        summary[mode]=metrics(pred)
        pr,rc,f1,sup=precision_recall_fscore_support(pred.truth,pred.prediction,labels=CODES,zero_division=0)
        for i,c in enumerate(CODES): classes.append(dict(mode=mode,error=c,precision=pr[i],recall=rc[i],f1=f1[i],n_views=int(sup[i]),n_reps=pred[pred.truth.eq(c)].rep_id.nunique()))
        cm=confusion_matrix(pred.truth,pred.prediction,labels=CODES)
        for i,t in enumerate(CODES):
            for j,h in enumerate(CODES): conf.append(dict(mode=mode,truth=t,prediction=h,count=int(cm[i,j])))
        for view,g in pred.groupby('view_bin',observed=True):
            views.append(dict(mode=mode,view=str(view),error='ALL',**metrics(g)))
            for c in CODES:
                sub=g[g.truth.eq(c)]
                views.append(dict(mode=mode,view=str(view),error=c,n_views=len(sub),n_reps=sub.rep_id.nunique(),accuracy=float(sub.truth.eq(sub.prediction).mean()) if len(sub) else None,macro_f1=None,balanced_accuracy=None))
        for threshold in [0.,.5,.7,.9]:
            keep=pred.max_probability.ge(threshold)
            correct=pred.truth.eq(pred.prediction)
            coverage.append(dict(mode=mode,threshold=threshold,n_total=len(pred),n_reported=int(keep.sum()),coverage=float(keep.mean()),reported_accuracy=float(correct[keep].mean()) if keep.any() else None,correct_reported_fraction=float((correct & keep).mean())))
    for name,records in [('folds',fold_rows),('per_class',classes),('confusion',conf),('by_view',views),('abstention',coverage)]:
        pd.DataFrame(records).to_csv(args.out_dir/(name+'.csv'),index=False)
    pd.concat(all_preds,ignore_index=True).to_csv(args.out_dir/'predictions.csv',index=False)
    report=dict(scope='GT Bad repetitions only; held-out persons; one view per inference; no deployed binary gate, counter, or Hailo evaluation',classes=CODES,n_bad_reps=278,n_bad_views=834,n_people=5,source_hashes={k:m[k] for k in ['features_sha256','repetitions_sha256']},annotation_sha256=sha(args.annotation),config=dict(C=.1,thresholds='exploratory only; not calibrated or selected',evaluation='full GT interval; augmentation training only',features='existing 180'),results=summary)
    (args.out_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n===== SUMMARY =====\n'+json.dumps(summary,indent=2))
    print('\n===== CLASS METRICS =====\n'+pd.DataFrame(classes).to_string(index=False))
    print('\n===== VIEW METRICS =====\n'+pd.DataFrame(views).query('error == "ALL"').to_string(index=False))
    print('\n===== ABSTENTION (exploratory) =====\n'+pd.DataFrame(coverage).to_string(index=False))
    print('Saved:',args.out_dir)

if __name__=='__main__': main()
