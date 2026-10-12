#!/usr/bin/env python3
"""REHAB24-6 Camera17 YOLO17 Squat correct/incorrect baseline.

Usage from ~/capstone2/pose_training:
    python scripts/train_squat_baseline.py

Expected input:
 data/processed/REHAB24-6/camera17_yolov8m_tracked/{index.csv,PM_*_rep*.npz}

Labels: 1=correct, 0=incorrect; evaluation splits by person_id (LOPO).
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_IN = Path('data/processed/REHAB24-6/camera17_yolov8m_tracked')
DEFAULT_OUT = Path('results/squat_baseline')
# COCO 17 indices
LS, RS, LH, RH, LK, RK, LA, RA = 5, 6, 11, 12, 13, 14, 15, 16


def distance(a, b):
    return np.linalg.norm(a-b, axis=-1)


def mid(a, b):
    # Treat missing points conservatively: one known side can form a center.
    both = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    only_a = np.isfinite(a).all(axis=-1) & ~np.isfinite(b).all(axis=-1)
    only_b = np.isfinite(b).all(axis=-1) & ~np.isfinite(a).all(axis=-1)
    out = np.full_like(a, np.nan)
    out[both] = (a[both] + b[both]) * 0.5
    out[only_a] = a[only_a]
    out[only_b] = b[only_b]
    return out


def angle(a, b, c):
    """Angle ABC in degrees, invalid geometry -> NaN."""
    u, v = a-b, c-b
    nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    den = nu*nv
    cos = np.full(len(den), np.nan)
    valid = np.isfinite(den) & (den > 1e-6)
    cos[valid] = np.sum(u[valid]*v[valid],axis=1) / den[valid]
    return np.degrees(np.arccos(np.clip(cos,-1,1)))


def stats(series, prefix):
    series = np.asarray(series, dtype=float)
    valid = series[np.isfinite(series)]
    keys = ['mean','std','min','max','q10','median','q90','range','first','last']
    if len(valid) == 0:
        return {f'{prefix}_{k}': np.nan for k in keys}
    # Percentages, not absolute frame counts, to avoid rep duration leakage.
    n = max(1, int(np.ceil(len(series) * 0.15)))
    def finite_mean(values):
        good = values[np.isfinite(values)]
        return float(good.mean()) if len(good) else np.nan
    vals = [float(valid.mean()),float(valid.std()),float(valid.min()),
            float(valid.max()),float(np.quantile(valid,0.10)),
            float(np.median(valid)),float(np.quantile(valid,0.90)),
            float(valid.max()-valid.min()),
            finite_mean(series[:n]),finite_mean(series[-n:])]
    return {f'{prefix}_{key}':value for key,value in zip(keys,vals)}


def extract_features(seq):
    """Repetition-level geometry summary; avoids absolute camera location/size."""
    if seq.ndim != 3 or seq.shape[1:] != (17,3) or len(seq) < 10:
        raise ValueError(f'Unexpected keypoint shape {seq.shape}')
    xy = np.array(seq[:,:,:2],dtype=float,copy=True)
    cf = np.asarray(seq[:,:,2], dtype=float)
    xy[(cf < 0.30) | ~np.isfinite(cf)] = np.nan

    shoulder = mid(xy[:,LS],xy[:,RS])
    hip = mid(xy[:,LH],xy[:,RH])
    ankle = mid(xy[:,LA],xy[:,RA])

    torso = distance(shoulder,hip)
    good_torso = torso[np.isfinite(torso) & (torso > 10)]
    scale = float(np.median(good_torso)) if len(good_torso) else np.nan
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Missing usable torso scale')
    norm = lambda d: d/scale

    left_knee = angle(xy[:,LH],xy[:,LK],xy[:,LA])
    right_knee = angle(xy[:,RH],xy[:,RK],xy[:,RA])
    left_hip = angle(xy[:,LS],xy[:,LH],xy[:,LK])
    right_hip = angle(xy[:,RS],xy[:,RH],xy[:,RK])

    # torso tilt relative to vertical; front and side camera interpretation differs.
    dx = np.abs(shoulder[:,0]-hip[:,0])
    dy = np.abs(shoulder[:,1]-hip[:,1])
    torso_tilt = np.degrees(np.arctan2(dx,dy))

    ankle_sep = np.abs(xy[:,LA,0]-xy[:,RA,0])
    knee_sep = np.abs(xy[:,LK,0]-xy[:,RK,0])
    # More stable than a ratio when the ankles overlap in half-profile view.
    data = {
        'left_knee_angle': left_knee,
        'right_knee_angle': right_knee,
        'knee_angle_mean': mid_scalar(left_knee,right_knee),
        'left_hip_angle': left_hip,
        'right_hip_angle': right_hip,
        'torso_tilt': torso_tilt,
        'hip_ankle_vertical': norm(ankle[:,1]-hip[:,1]),
        'hip_knee_vertical_left': norm(xy[:,LK,1]-xy[:,LH,1]),
        'hip_knee_vertical_right': norm(xy[:,RK,1]-xy[:,RH,1]),
        'knee_ankle_horizontal_left': norm(xy[:,LK,0]-xy[:,LA,0]),
        'knee_ankle_horizontal_right': norm(xy[:,RK,0]-xy[:,RA,0]),
        'knee_separation': norm(knee_sep),
        'ankle_separation': norm(ankle_sep),
        'knee_minus_ankle_separation': norm(knee_sep-ankle_sep),
        'hip_relative_to_ankles_x': norm(hip[:,0]-ankle[:,0]),
    }
    # Squat depth (vertical hip travel) relative to mean of first 15% frames.
    valid_hip = hip[:,1][np.isfinite(hip[:,1])]
    if len(valid_hip):
        n = max(1,int(np.ceil(len(hip)*0.15)))
        ref = hip[:n,1]
        ref = ref[np.isfinite(ref)]
        if len(ref):
            data['hip_y_from_start'] = norm(hip[:,1]-float(ref.mean()))

    result = {}
    for name,values in data.items():
        result.update(stats(values,name))
    return result


def mid_scalar(a,b):
    a,b = np.asarray(a),np.asarray(b)
    return np.where(np.isfinite(a)&np.isfinite(b), (a+b)/2,
                    np.where(np.isfinite(a),a,b))


def score(y_true,y_pred):
    return {
        'accuracy': float(accuracy_score(y_true,y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true,y_pred)),
        'incorrect_precision': float(precision_score(y_true,y_pred,pos_label=0,zero_division=0)),
        'incorrect_recall': float(recall_score(y_true,y_pred,pos_label=0,zero_division=0)),
        'incorrect_f1': float(f1_score(y_true,y_pred,pos_label=0,zero_division=0)),
        'macro_f1': float(f1_score(y_true,y_pred,average='macro',zero_division=0)),
    }


def build_models():
    return {
        'RandomForest': Pipeline([
            ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('model',RandomForestClassifier(n_estimators=300,max_depth=6,
                 min_samples_leaf=3,class_weight='balanced_subsample',random_state=42,n_jobs=-1)),
        ]),
        'LogisticRegression': Pipeline([
            ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('scaler',StandardScaler()),
            ('model',LogisticRegression(class_weight='balanced',max_iter=3000,
                                        random_state=42,C=0.1)),
        ]),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=DEFAULT_IN)
    ap.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    index=args.data_dir/'index.csv'
    if not index.is_file():
        raise FileNotFoundError(index)
    df=pd.read_csv(index)
    necessary=['file','person_id','label','video_id','repetition_number']
    miss=[k for k in necessary if k not in df.columns]
    if miss: raise ValueError(f'Missing index columns {miss}: {df.columns.tolist()}')
    if df.duplicated(['video_id','repetition_number']).any():
        raise ValueError('Duplicated repetition rows')
    print('Index rows:',len(df),'Unique people:',df.person_id.nunique())
    print('Class distribution (1=correct, 0=incorrect):',df.label.value_counts().to_dict())
    if len(df)!=195: warnings.warn(f'Expected 195 repetitions; found {len(df)}')
    if 'target_found_frames' in df.columns:
        coverage=(df.target_found_frames/df.frames)
        print(f'Subject detected frame ratio: mean={coverage.mean():.3f}; min={coverage.min():.3f}')
        print('Samples below 90% detection:',int((coverage<0.90).sum()))
    if 'valid_keypoint_ratio' in df.columns:
        print(f'Valid keypoint ratio: mean={df.valid_keypoint_ratio.mean():.3f}; min={df.valid_keypoint_ratio.min():.3f}')
    if 'max_hip_jump_px' in df.columns:
        print('Max interframe hip jump (px):',df.max_hip_jump_px.max())

    rows=[]; bad=[]; keep=[]
    for idx,r in df.iterrows():
        try:
            with np.load(args.data_dir/str(r['file']),allow_pickle=False) as item:
                if int(item['label'])!=int(r.label) or int(item['person_id'])!=int(r.person_id):
                    raise ValueError('Index-label mismatch')
                feats=extract_features(item['keypoints'])
            rows.append(feats); keep.append(idx)
        except Exception as e:
            bad.append((r['file'],str(e)))
    if bad:
        print('\nExcluded invalid repetitions:')
        for item in bad:print(item)
    if len(keep) < 30: raise RuntimeError('Not enough valid repetitions')
    df=df.loc[keep].reset_index(drop=True)
    features=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    empty=features.columns[features.isna().all()].tolist()
    if empty:
        print('Dropping globally all-NaN features:',empty)
        features=features.drop(columns=empty)
    if features.isna().all(axis=1).any():
        raise RuntimeError('At least one repetition has no valid features')
    X=features.to_numpy(dtype=float)
    y=df.label.to_numpy(dtype=int)
    groups=df.person_id.to_numpy()
    assert set(np.unique(y))=={0,1}, 'Both labels are required'
    assert len(np.unique(groups))>=3, 'Need multiple held-out participants'
    print('Valid samples:',len(df),'Features:',X.shape[1])
    args.out_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.out_dir/'used_repetitions.csv',index=False)
    features.to_csv(args.out_dir/'features.csv',index=False)
    logo=LeaveOneGroupOut()
    templates=build_models()
    report={
        'data':'REHAB24-6 Camera17, YOLOv8m-pose tracked, Squat, repetition-level',
        'label_convention':{'0':'incorrect','1':'correct'},
        'validation':'Leave-One-Person-Out (9 person IDs)',
        'n_repetitions':int(len(y)),
        'n_people':int(df.person_id.nunique()),
        'class_counts':{str(v):int((y==v).sum()) for v in [0,1]},
        'feature_count':int(X.shape[1]),
        'results':{},
    }
    results=[]
    all_preds={}
    for model_name,template in templates.items():
        preds=np.full(len(y),-1,dtype=int)
        fold_rows=[]
        for train,test in logo.split(X,y,groups):
            assert not (set(groups[train]) & set(groups[test]))
            model=clone(template)
            model.fit(X[train],y[train])
            preds[test]=model.predict(X[test])
            fm=score(y[test],preds[test]);fm['person_id']=int(groups[test[0]])
            fm['test_repetitions']=int(len(test))
            fm['test_incorrect']=int((y[test]==0).sum())
            fold_rows.append(fm)
        assert (preds==-1).sum()==0
        overall=score(y,preds)
        overall['fold_mean_balanced_accuracy']=float(np.mean([f['balanced_accuracy'] for f in fold_rows]))
        report['results'][model_name]={'pooled_out_of_person':overall,'per_person':fold_rows}
        print(f'\n{model_name}: LOPO out-of-person results')
        for key,val in overall.items():print(f'  {key}: {val:.4f}')
        all_preds[model_name]=preds
        results.append((model_name,overall['balanced_accuracy']))
        pd.DataFrame(fold_rows).to_csv(args.out_dir/f'{model_name}_per_person.csv',index=False)
    results.sort(key=lambda x:x[1],reverse=True)
    best_name=results[0][0]
    report['selected_model']=best_name
    # The final model is fitted to *all* subjects after out-of-person evaluation.
    final=clone(templates[best_name]).fit(X,y)
    joblib.dump({'pipeline':final,'feature_names':features.columns.tolist(),
                 'label_map':{0:'incorrect',1:'correct'},
                 'source':'REHAB24-6 YOLOv8m pose tracked camera17'},
                args.out_dir/'squat_classifier.joblib')
    predictions=df[['video_id','person_id','repetition_number','label']].copy()
    for name,pred in all_preds.items():predictions['pred_'+name]=pred
    predictions.to_csv(args.out_dir/'lopo_predictions.csv',index=False)
    cm=confusion_matrix(y,all_preds[best_name],labels=[1,0])
    report['best_model_confusion_matrix_labels']=['correct','incorrect']
    report['best_model_confusion_matrix']=cm.tolist()
    fig,ax=plt.subplots(figsize=(5.2,4.5))
    im=ax.imshow(cm)
    ax.set_xticks([0,1],labels=['Correct','Incorrect'])
    ax.set_yticks([0,1],labels=['Correct','Incorrect'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground truth')
    ax.set_title(f'{best_name}: Leave-One-Person-Out')
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha='center',va='center',color='white' if v > cm.max()/2 else 'black',fontsize=13)
    fig.tight_layout();fig.savefig(args.out_dir/'confusion_matrix.png',dpi=180);plt.close(fig)
    with (args.out_dir/'metrics.json').open('w') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('\nChosen final model:',best_name)
    print('Saved outputs to:',args.out_dir.resolve())
    print('NOTE: LOPO scores are for held-out people; final joblib is refit on all 9 people.')

if __name__=='__main__':
    main()
