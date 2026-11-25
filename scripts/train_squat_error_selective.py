#!/usr/bin/env python3
"""Selective squat error detection with Good negatives and nested person-disjoint validation.
Uses existing YOLO caches; no changes to runtime or deployed Good/Bad model.
Viewing angle is annotation-only for this experiment, not estimated at runtime.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from train_squat_error_geometry import geometry
from train_squat_error_baseline import sha, CODES

SPECS = {
    'ERR_KNEES_INSIDE': (('knee_width','ankle_width','knee_minus_ankle_width','knee_ankle_ratio','inward_l','inward_r'), 'frontal'),
    'ERR_TORSO_TO_SIDE': (('torso_x_abs','torso_x_signed','shoulder_level_abs','hip_level_abs'), 'frontal'),
    'ERR_SHALLOW_SQUAT': (('hip_height_rel_ankle','knee_angle_l','knee_angle_r','hip_knee_y_l','hip_knee_y_r'), 'oblique_profile'),
}
THRESHOLDS = [.5,.6,.7,.8,.9,.95,.97,.99]
TARGET_PRECISION = .85
MIN_POSITIVE_REPS = 8
MIN_POSITIVE_PERSONS = 3

def make_model():
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                         StandardScaler(),
                         LogisticRegression(C=.1, class_weight='balanced',
                                            max_iter=3000, random_state=42))

def evaluate(d, flag):
    y = d.truth.eq(d.error).to_numpy()
    good = d.label.eq(1).to_numpy()
    flag = np.asarray(flag, bool)
    tp = int((flag & y).sum())
    fp = int((flag & ~y).sum())
    badfp = int((flag & ~y & ~good).sum())
    def ratio(a,b): return float(a/b) if b else None
    return dict(n_views=len(d), n_reps=int(d.rep_id.nunique()), positives=int(y.sum()),
                alerts=int(flag.sum()), tp=tp, fp=fp,
                precision=ratio(tp,tp+fp), recall=ratio(tp,int(y.sum())),
                conditional_bad_precision=ratio(tp,tp+badfp),
                good_views=int(good.sum()), good_false_alerts=int((flag & good).sum()),
                good_false_alert_rate=ratio(int((flag & good).sum()),int(good.sum())),
                correct_positive_reps=int(d.loc[flag & y,'rep_id'].nunique()),
                correct_positive_persons=int(d.loc[flag & y,'person_id'].nunique()))

def choose_threshold(d, scores):
    # Threshold chosen using inner held-out predictions, never outer test data.
    choices=[]
    for th in THRESHOLDS:
        m=evaluate(d,scores>=th)
        if (m['precision'] is not None and m['precision']>=TARGET_PRECISION
            and m['correct_positive_reps']>=MIN_POSITIVE_REPS
            and m['correct_positive_persons']>=MIN_POSITIVE_PERSONS):
            choices.append((m['recall'],m['precision'],th))
    return max(choices)[2] if choices else None

def load_geometry(args):
    b=args.build_dir
    m=json.loads((b/'manifest.json').read_text())
    for file,key in [('features_crop10.npy','features_sha256'),('repetitions.csv','repetitions_sha256')]:
        assert sha(b/file)==m[key], f'Hash mismatch: {file}'
    ep=Path('data/preprocess/extra_datasets/manifest.json')
    assert sha(ep)==m['rehab26_manifest_sha256']
    em=json.loads(ep.read_text())
    assert em['weights_sha256']==m['pose_weights_sha256']
    d=pd.read_csv(b/'repetitions.csv')
    d['row']=np.arange(len(d))
    d=d[d.dataset.eq('rehab26')].copy().reset_index(drop=True)
    assert len(d)==1662 and d.rep_id.nunique()==554 and d.person_id.nunique()==5
    assert not d.duplicated(['rep_id','camera']).any()
    assert (d.groupby('rep_id').camera.nunique()==3).all()
    assert (d.groupby('rep_id').person_id.nunique()==1).all()
    a=pd.read_csv(args.annotation,sep=';')
    a=a[a.exercise_id.eq(6)].copy()
    assert len(a)==554 and set(a.correctness_score)=={1,5}
    assert a.additional_errors.fillna('').str.strip().eq('').all()
    a['rep_id']='rehab26:'+a.video_id.astype(str)+':'+a.person_orientation.astype(int).astype(str)+':'+a.repetition_number.astype(int).astype(str)
    assert not a.rep_id.duplicated().any() and set(a.rep_id)==set(d.rep_id)
    a=a.set_index('rep_id')
    for cached,raw in [('person_id','person'),('first_frame','first_frame'),('last_frame','last_frame'),('orientation','person_orientation'),('error_description','error_description')]:
        assert np.array_equal(d[cached].fillna('').to_numpy(), d.rep_id.map(a[raw]).fillna('').to_numpy()), cached
    assert np.array_equal(d.label.to_numpy(), d.rep_id.map(a.correctness_score).eq(1).astype(int).to_numpy())
    d['truth']=d.error_description.fillna('GOOD')
    assert set(d.truth)==set(CODES)|{'GOOD'}
    assert d.viewing_angle_deg.notna().all() and d.viewing_angle_deg.abs().le(90).all()
    import preprocess_rehab24 as pre
    from extract_video_kp_cache import KP_DUMP_DTYPE
    from build_squat_multiview import assess
    vids={v['sample_id']:v for v in em['videos'] if v['dataset']=='rehab26' and int(v['exercise_id'])==6}
    assert len(vids)==15
    records={}
    for sid, rows in d.groupby('sample_id'):
        v=vids[sid]
        assert pre.check_cache(v,m['pose_weights_sha256'])=='complete'
        assert abs(v['fps']-30)<.01
        paths=pre.cache_paths(v)
        k=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r')
        assert np.array_equal(k['frame_id'],np.arange(len(k)))
        with np.load(paths[2],allow_pickle=False) as z:
            status=z['status']; boxes=z['selected_box']
        for idx,r in rows.iterrows():
            lo,hi=int(r.first_frame),int(r.last_frame)
            assert 0<=lo<=hi<len(k)
            seq,_,why=assess(k['kp'],status,boxes,lo,hi)
            if why: raise ValueError(f'Previously accepted row rejected: {sid}, {lo}, {why}')
            records[idx]=geometry(seq)
        print('Features',sid,len(rows),flush=True)
    g=pd.DataFrame.from_dict(records,orient='index').sort_index()
    assert g.index.tolist()==list(range(len(d))) and g.shape==(1662,118)
    assert not np.isinf(g.to_numpy()).any()
    return d,g

def experiment(d,g):
    predictions=[]; folds=[]; summary=[]
    for error,(prefixes,view) in SPECS.items():
        mask=(d.viewing_angle_deg.abs().le(30) if view=='frontal'
              else d.viewing_angle_deg.abs().gt(30))
        z=d.loc[mask].copy().reset_index(drop=True)
        z['error']=error; z['view_policy']=view
        names=[c for c in g.columns if any(c.startswith(p+'_') for p in prefixes)]
        x=g.loc[mask,names].to_numpy()
        y=z.truth.eq(error).to_numpy()
        parts=[]
        for person in sorted(z.person_id.unique()):
            tr=z.person_id.ne(person).to_numpy(); te=~tr
            assert set(z.loc[tr,'rep_id']).isdisjoint(z.loc[te,'rep_id'])
            inner=np.full(len(z),np.nan)
            for val in sorted(z.loc[tr,'person_id'].unique()):
                it=tr & z.person_id.ne(val).to_numpy()
                iv=tr & z.person_id.eq(val).to_numpy()
                model=make_model().fit(x[it],y[it])
                inner[iv]=model.predict_proba(x[iv])[:,1]
            assert np.isfinite(inner[tr]).all()
            threshold=choose_threshold(z.loc[tr],inner[tr])
            model=make_model().fit(x[tr],y[tr])
            q=z.loc[te].copy()
            q['score']=model.predict_proba(x[te])[:,1]
            q['nested_threshold']=threshold
            q['alert_nested']=False if threshold is None else q.score.ge(threshold)
            parts.append(q)
            f=dict(error=error,view_policy=view,heldout_person=person,
                   threshold=threshold,**evaluate(q,q.alert_nested))
            folds.append(f)
            print(f,flush=True)
        q=pd.concat(parts,ignore_index=True)
        predictions.append(q)
        for label,flag in [('fixed_0.5',q.score.ge(.5)),('fixed_0.7',q.score.ge(.7)),
                           ('fixed_0.9',q.score.ge(.9)),('nested',q.alert_nested)]:
            summary.append(dict(error=error,view_policy=view,policy=label,n_features=len(names),**evaluate(q,flag)))
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(folds),pd.DataFrame(summary)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,default=Path('results/squat_multiview/build'))
    p.add_argument('--annotation',type=Path,default=Path.home()/'capstone2/datasets/REHAB26-ViewAngles/Segmentation.csv')
    p.add_argument('--out-dir',type=Path,default=Path('results/squat_error_selective'))
    args=p.parse_args()
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    d,g=load_geometry(args)
    pred,folds,summary=experiment(d,g)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    pred.to_csv(args.out_dir/'predictions.csv',index=False)
    folds.to_csv(args.out_dir/'folds.csv',index=False)
    summary.to_csv(args.out_dir/'summary.csv',index=False)
    info=dict(scope='GT full repetition; Good negatives included; annotated viewing angle; exploratory on same five participants; no deployed Good/Bad gate', target_precision=TARGET_PRECISION,min_positive_reps=MIN_POSITIVE_REPS,min_positive_persons=MIN_POSITIVE_PERSONS,threshold_grid=THRESHOLDS,specs=SPECS)
    (args.out_dir/'config.json').write_text(json.dumps(info,indent=2))
    print(summary.to_string(index=False)); print('Saved:',args.out_dir)

if __name__=='__main__': main()
