#!/usr/bin/env python3
"""Exploratory 9-fold person-held-out comparison: feature group and mild temporal cropping.

No change to streaming detection. Each fold trains on eight OTHER people only.
Events supply only automatically detected boundaries; predictions in events are NEVER
used as training labels or evaluation targets. Held-out evaluation reuses saved GT
keypoints and clips them to event start/end (intersection; NOT full online buffer).
This is NOT a fresh external test or an online stream run.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

VIDEOS = ['PM_008','PM_022','PM_029','PM_038','PM_043','PM_105','PM_113','PM_118','PM_126']
VARIANTS = (
    ('Baseline160', 'base', 0.0),
    ('PlusAsymmetry180', 'base_asym', 0.0),
    ('PlusAsymmetry180_crop05', 'base_asym', 0.05),
    ('PlusAsymmetry180_crop10', 'base_asym', 0.10),
)


def trim_indices(n, fraction):
    """Deterministic four sequences: full, trim start, trim end, trim both."""
    if fraction == 0:
        return [(0, n)]
    if not 0 < fraction <= 0.2:
        raise ValueError('Trimming fraction must be within (0,.2]')
    cut = max(1, round(n * fraction))
    if n - 2 * cut < 10:
        raise ValueError(f'Sequence of length {n} is too short for crop {fraction}')
    return [(0, n), (cut, n), (0, n-cut), (cut, n-cut)]


def overlap_iou(a0, a1, b0, b1):
    common = max(0, min(a1,b1)-max(a0,b0)+1)
    union = (a1-a0+1) + (b1-b0+1) - common
    return common/union if union else 0.0


def match_events(samples, events, threshold=.2):
    """Greedy one-to-one boundary matching, independent of posture predictions."""
    candidates = sorted((overlap_iou(int(s['ids'][0]), int(s['ids'][-1]),
                                    int(e['start_frame']), int(e['end_frame'])), i, j)
                         for i, s in enumerate(samples) for j, e in enumerate(events))
    matching = {}
    used = set()
    for overlap,i,j in reversed(candidates):
        if overlap < threshold: break
        if i not in matching and j not in used:
            matching[i] = events[j]
            used.add(j)
    return matching


def stats(y, pred):
    y = np.asarray(y,dtype=int)
    pred = np.asarray(pred,dtype=int)
    if len(y) != len(pred): raise ValueError('Length mismatch')
    n_good = int((y==1).sum()); n_bad=int((y==0).sum())
    return {'n':len(y), 'accuracy':float((y==pred).mean()),
            'good_correct':int(((y==1)&(pred==1)).sum()), 'good_total':n_good,
            'bad_correct':int(((y==0)&(pred==0)).sum()), 'bad_total':n_bad,
            'false_alarm':int(((y==1)&(pred==0)).sum()),
            'missed_bad':int(((y==0)&(pred==1)).sum())}


def extract_vector(seq, names, extract_feature_groups):
    groups=extract_feature_groups(seq)
    features={**groups['base'],**groups['asymmetry']}
    if not set(names).issubset(features):
        raise ValueError('Unexpected feature names')
    return np.array([features[n] for n in names], dtype=float)


def run(args):
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from train_squat_extended import extract_feature_groups

    reference=joblib.load(args.template_model)  # Load ONLY own model artifact.
    if reference.get('feature_variant')!='PlusAsymmetry180':
        raise ValueError('Expected PlusAsymmetry180 template')
    names_all=reference['feature_names']
    if len(names_all)!=180: raise ValueError('Expected 180 named features')
    index=pd.read_csv(args.data_dir/'index.csv')
    if set(index.video_id.unique()) != set(VIDEOS):
        raise ValueError('Expected exact nine video IDs')
    if index.person_id.nunique()!=9 or len(index)!=195:
        raise ValueError('Expected nine subjects and 195 repetitions')
    if index.duplicated(['video_id','repetition_number']).any():
        raise ValueError('Duplicate samples')
    if (index.groupby('person_id').video_id.nunique()!=1).any() or (index.groupby('video_id').person_id.nunique()!=1).any():
        raise ValueError('Expected one video per subject')
    if not set(index.label.unique()) <= {0,1}:
        raise ValueError('Expected labels 0/1')

    samples=[]
    for r in index.itertuples(index=False):
        with np.load(args.data_dir/str(r.file),allow_pickle=False) as z:
            kp=np.array(z['keypoints'],copy=True)
            ids=np.asarray(z['frame_indices'],dtype=int)
            yy=int(np.asarray(z['label']).item())
            pp=int(np.asarray(z['person_id']).item())
        if len(kp)!=len(ids) or len(kp)<15 or yy!=int(r.label) or pp!=int(r.person_id):
            raise ValueError(f'NPZ mismatch: {r.file}')
        samples.append({'video':str(r.video_id),'person':pp,'rep':int(r.repetition_number),
                        'label':yy,'kp':kp,'ids':ids})

    # Online events used solely for interval boundaries. No event correctness/posture labels are read.
    byvideo=defaultdict(list)
    for i,s in enumerate(samples): byvideo[s['video']].append(i)
    for video,inds in byvideo.items():
        path=args.events_dir/f'{video}_events.jsonl'
        with path.open(encoding='utf-8') as f:
            events=[json.loads(line) for line in f if line.strip()]
        events=[e for e in events if e.get('type')=='rep']
        local=[samples[i] for i in inds]
        matched=match_events(local,events)
        if len(matched)!=len(inds) or len(matched)!=len(events):
            raise ValueError(f'{video}: expected a full one-to-one event match; got '
                             f'{len(matched)} matched for {len(inds)} GT, {len(events)} events')
        for i,e in matched.items():
            s=local[i]
            mask=(s['ids']>=int(e['start_frame'])) & (s['ids']<=int(e['end_frame']))
            if mask.sum()<10:
                raise ValueError(f'{video} rep{s["rep"]}: intersection has <10 frames')
            s['cropped_kp']=s['kp'][mask]
            s['retained']=float(mask.mean())

    # Cache extraction, avoid re-extracting identical sequences for each fold.
    feats_full={}; feats_crop={}; augcache={}
    full_feature_groups=extract_feature_groups(samples[0]['kp'])
    base_names=[n for n in names_all if n in full_feature_groups['base']]
    if len(base_names)!=160: raise ValueError('Expected 160 baseline features')
    all_names={'base':base_names,'base_asym':list(names_all)}
    for i,s in enumerate(samples):
        feats_full[i]=extract_vector(s['kp'],names_all,extract_feature_groups)
        feats_crop[i]=extract_vector(s['cropped_kp'],names_all,extract_feature_groups)
        for frac in (0.05,0.10):
            slices=trim_indices(len(s['kp']),frac)
            augcache[i,frac]=[feats_full[i]] + [extract_vector(s['kp'][st:ed],names_all,extract_feature_groups)
                                                 for st,ed in slices[1:]]

    outrows=[]; foldrows=[]; save_paths=[]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    # Fixed family and augmentations chosen as EXPLORATORY candidates, no tuning on fold's held-out labels.
    for variant, group, fraction in VARIANTS:
        names=all_names[group]
        positions=np.array([names_all.index(n) for n in names],dtype=int)
        for video in VIDEOS:
            test=[i for i,s in enumerate(samples) if s['video']==video]
            train=[i for i,s in enumerate(samples) if s['video']!=video]
            train_people=set(samples[i]['person'] for i in train)
            test_people=set(samples[i]['person'] for i in test)
            assert not (train_people & test_people), 'LOPO subject leakage'
            X=[]; Y=[]
            for i in train:
                vs=[feats_full[i]] if fraction==0 else augcache[i,fraction]
                for v in vs:
                    X.append(v[positions]);Y.append(samples[i]['label'])
            model=clone(reference['pipeline'])
            model.fit(np.stack(X),np.asarray(Y))
            Xfull=np.stack([feats_full[i][positions] for i in test])
            Xcrop=np.stack([feats_crop[i][positions] for i in test])
            fullpred=model.predict(Xfull).astype(int)
            croppred=model.predict(Xcrop).astype(int)
            lab=np.array([samples[i]['label'] for i in test],dtype=int)
            fullmetrics=stats(lab,fullpred); cropmetrics=stats(lab,croppred)
            foldrows.append({'variant':variant,'video':video,'features':len(names),
                             'train_reps':len(train),'train_feature_rows':len(X),
                             'test_reps':len(test), 'avg_retained':float(np.mean([samples[i]['retained'] for i in test])),
                             'full_accuracy':fullmetrics['accuracy'], 'cropped_accuracy':cropmetrics['accuracy'],
                             'full_bad_correct':fullmetrics['bad_correct'],'crop_bad_correct':cropmetrics['bad_correct'],
                             'full_false_alarm':fullmetrics['false_alarm'],'crop_false_alarm':cropmetrics['false_alarm']})
            for ix,i in enumerate(test):
                outrows.append({'variant':variant,'video':video,'rep':samples[i]['rep'],'gt':int(lab[ix]),
                                'offline_full_pred':int(fullpred[ix]),'offline_cropped_pred':int(croppred[ix]),
                                'retained_gt_fraction':samples[i]['retained']})
            # Save fitted classifier weights only when explicitly requested.
            if args.save_models:
                path=args.out_dir/'models'/variant/f'{video}.joblib'
                path.parent.mkdir(parents=True,exist_ok=True)
                joblib.dump({'pipeline':model,'feature_names':list(names),'feature_variant':variant,
                             'held_out_video':video,'training_person_ids':sorted(train_people),
                             'note':'Exploratory person-held-out fit; manual-segment training data and preselected hyperparameters.'},path)
                save_paths.append(str(path))
            print(f'{variant:27s} {video} full={fullmetrics["accuracy"]:.3f} '
                  f'crop={cropmetrics["accuracy"]:.3f} '
                  f'crop_bad={cropmetrics["bad_correct"]}/{cropmetrics["bad_total"]} '
                  f'crop_false_alarm={cropmetrics["false_alarm"]}',flush=True)
    preds=pd.DataFrame(outrows)
    folds=pd.DataFrame(foldrows)
    preds.to_csv(args.out_dir/'per_rep.csv',index=False)
    folds.to_csv(args.out_dir/'per_video.csv',index=False)
    summary=[]
    for variant,grp,frac in VARIANTS:
        v=preds[preds.variant==variant]
        if len(v)!=195 or len(v[['video','rep']].drop_duplicates())!=195:
            raise RuntimeError('Prediction count mismatch')
        for col,setting in [('offline_full_pred','full'),('offline_cropped_pred','cropped')]:
            m=stats(v["gt"],v[col])
            summary.append({'variant':variant,'setting':setting,'feature_count':len(all_names[grp]),
                            'train_trim_fraction':frac,**m})
    summary_df=pd.DataFrame(summary)
    summary_df.to_csv(args.out_dir/'summary.csv',index=False)
    (args.out_dir/'metadata.json').write_text(json.dumps({
        'videos':VIDEOS,'source':'REHAB24-6 tracked COCO17 GT rep NPZ',
        'evaluation':'held-out GT full or intersection with detected online event intervals',
        'caveats':['Offline cropped input is NOT full online buffered keypoints.',
                   'Fold classification weights exclude test person, but feature choice and detector thresholds were explored on all nine people.',
                   'Train-crop augmentation can change the meaning of a repetition-level label when a fault occurs only in a discarded phase.',
                   'Variants and trimming rates are exploratory; choosing based on these test folds makes final accuracy optimistic.'],
        'saved_models':save_paths,
    },indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n===== POOLED OFFLINE LOPO COMPARISON =====')
    print(summary_df[['variant','setting','accuracy','bad_correct','bad_total','false_alarm','missed_bad']].to_string(index=False))
    print(f'\nSaved: {args.out_dir}/summary.csv /per_video.csv /per_rep.csv')
    print('EXPLORATORY: these 9 people were already used in feature/hyperparameter design; no new-user guarantee.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'))
    ap.add_argument('--events-dir',type=Path,default=Path('results/stream_lopo'))
    ap.add_argument('--template-model',type=Path,default=Path('models/squat_plus_asymmetry180.joblib'))
    ap.add_argument('--out-dir',type=Path,default=Path('results/lopo_robustness_experiment'))
    ap.add_argument('--save-models',action='store_true',help='Save fold models for future actual video streaming; OFF by default.')
    args=ap.parse_args()
    run(args)

if __name__=='__main__': main()
