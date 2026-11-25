#!/usr/bin/env python3
"""Train frozen LR candidates for the nine REHAB24 Camera17 squat subjects.
Reuse baseline feature arrays; exclude each person's ALL REHAB24 windows.
Run from pose_training_offline. No GPU inference or raw cache changes.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
from threadpoolctl import threadpool_limits
import train_exercise_baseline as baseline

LABELS = list(baseline.LABELS)
TARGETS = {'PM_008': '1', 'PM_022': '2', 'PM_029': '3',
           'PM_038': '4', 'PM_043': '5', 'PM_105': '6',
           'PM_126': '7', 'PM_113': '8', 'PM_118': '9'}
COLS = ['sample_id', 'dataset', 'person_id', 'group', 'exercise',
        'camera', 'label', 'first_frame', 'last_frame']
KEYS = ['sample_id', 'first_frame', 'last_frame']


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def load_inputs(root, seconds):
    p = root / f'{seconds}s'
    paths = {k: p / k for k in ('features.npy', 'windows.csv', 'oof_predictions.csv')}
    X = np.load(paths['features.npy'], allow_pickle=False)
    df = pd.read_csv(paths['windows.csv'], dtype=str)
    oof = pd.read_csv(paths['oof_predictions.csv'], dtype=str)
    df = df[COLS].copy()
    oof = oof.loc[(oof.scope == 'pooled') & (oof.model == 'LR'), COLS].copy()
    if X.shape != (len(df), 156) or not np.isfinite(X).all():
        raise ValueError(f'{seconds}s: invalid feature shape or values: {X.shape}')
    if df.isna().any().any() or oof.isna().any().any():
        raise ValueError('Missing window metadata')
    if df.duplicated(KEYS).any() or oof.duplicated(KEYS).any():
        raise ValueError('Duplicate window keys')
    if not df.sort_values(KEYS).reset_index(drop=True).equals(oof.sort_values(KEYS).reset_index(drop=True)):
        raise ValueError(f'{seconds}s: windows.csv differs from completed pooled LR evaluation')
    if set(df.label) != set(LABELS):
        raise ValueError('Unexpected training classes')
    # Check all known camera views and all rows in the person/group relationship.
    r = df[df.dataset.eq('rehab24')]
    for vid, pid in TARGETS.items():
        g = f'rehab24:{pid}'
        by_person = r.person_id.eq(pid)
        if not by_person.equals(r.group.eq(g)) or not by_person.any():
            raise ValueError(f'Invalid mapping for {vid} -> {g}')
        for cam in (17, 18):
            sid = f'Ex6_Camera{cam}_{vid}'
            a = r[r.sample_id.eq(sid)]
            if a.empty or set(a.person_id) != {pid} or set(a.group) != {g} or set(a.label) != {'Squat'}:
                raise ValueError(f'Invalid target mapping: {sid}')
    return X, df, {k: {'path': str(v), 'sha256': sha(v)} for k, v in paths.items()}


def counts(df):
    return {str(k): int(v) for k,v in df.label.value_counts().items()}


def train_one(X, df, seconds, vid, pid, out, sources):
    group = f'rehab24:{pid}'
    test = df.group.eq(group).to_numpy()
    tr, te = np.flatnonzero(~test), np.flatnonzero(test)
    train, hold = df.iloc[tr], df.iloc[te]
    if not len(te) or group in set(train.group) or ((train.dataset == 'rehab24') & (train.person_id == pid)).any():
        raise RuntimeError(f'Leakage guard failed: {vid}')
    if set(train.label) != set(LABELS):
        raise ValueError(f'Missing train classes: {vid}')
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', C=1))
    pipeline.fit(X[tr], train.label.to_numpy())
    pred = pipeline.predict(X[te])
    probabilities = pipeline.predict_proba(X[te])
    classes = pipeline.classes_.tolist()
    model_path = out / f'{vid}.joblib'
    baseline_path = Path(baseline.__file__)
    meta = {'schema_version': 1, 'window_seconds': seconds, 'sampling_fps': 15,
            'feature_dim': 156, 'heldout_video_id': vid, 'heldout_person_id': pid,
            'heldout_group': group, 'labels': LABELS, 'probability_classes': classes,
            'training_windows': len(tr), 'heldout_windows': len(te),
            'training_counts': counts(train), 'heldout_counts': counts(hold),
            'training_groups': sorted(train.group.unique().tolist()),
            'heldout_sample_ids': sorted(hold.sample_id.unique().tolist()),
            'train_test_group_overlap': [], 'inputs': sources,
            'feature_source_sha256': sha(baseline_path),
            'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                         'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}}
    joblib.dump({'pipeline': pipeline, 'metadata': meta}, model_path, compress=3)
    reloaded = joblib.load(model_path)['pipeline']
    np.testing.assert_array_equal(reloaded.predict(X[te]), pred)
    max_diff = float(np.max(np.abs(reloaded.predict_proba(X[te]) - probabilities)))
    if max_diff > 1e-12:
        raise RuntimeError(f'Reload mismatch: {vid}')
    z = hold.copy()
    z['prediction'] = pred
    for i, label in enumerate(classes):
        z[f'prob_{label}'] = probabilities[:, i]
    z['heldout_video_id'] = vid
    z['window_seconds'] = seconds
    cam = z.sample_id.eq(f'Ex6_Camera17_{vid}')
    metrics = {'window_seconds': seconds, 'video_id': vid, 'heldout_group': group,
               'train_windows': len(tr), 'heldout_windows': len(te),
               'train_groups': train.group.nunique(), 'heldout_group_overlap': 0,
               'heldout_accuracy': float((z.label == z.prediction).mean()),
               'heldout_macro_f1_present_classes': float(f1_score(z.label, z.prediction,
                   labels=[l for l in LABELS if l in set(z.label)], average='macro', zero_division=0)),
               'heldout_confusion': confusion_matrix(z.label, z.prediction, labels=LABELS).tolist(),
               'camera17_squat_windows': int(cam.sum()),
               'camera17_squat_correct': int((cam & z.prediction.eq('Squat')).sum()),
               'camera17_squat_recall': float(z.loc[cam, 'prediction'].eq('Squat').mean()),
               'reload_max_probability_difference': max_diff,
               'model_sha256': sha(model_path)}
    dump(out / f'{vid}.json', {**meta, 'evaluation': metrics})
    print(f'{seconds}s {vid} [{group}] train={len(tr)} heldout={len(te)} '
          f'overlap=0 Camera17 squat={metrics["camera17_squat_correct"]}/{metrics["camera17_squat_windows"]} reload_diff={max_diff:g}', flush=True)
    return metrics, z


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=int, choices=[2, 4], action='append')
    p.add_argument('--baseline-dir', type=Path, default=Path('results/exercise_baseline'))
    p.add_argument('--out-dir', type=Path, default=Path('models/exercise_lopo'))
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.threads < 1: p.error('--threads must be positive')
    seconds_list = sorted(set(a.seconds or [2, 4]))
    if a.out_dir.exists():
        raise FileExistsError(f'{a.out_dir} already exists; choose a new --out-dir (no overwrite)')
    inputs = {s: load_inputs(a.baseline_dir, s) for s in seconds_list}
    a.out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.exercise_lopo_', dir=a.out_dir.parent))
    results = []; predictions = []
    try:
        with threadpool_limits(limits=a.threads):
            for s, (X, df, sources) in inputs.items():
                out = stage / f'{s}s'; out.mkdir()
                for vid, pid in TARGETS.items():
                    m, z = train_one(X, df, s, vid, pid, out, sources)
                    results.append(m); predictions.append(z)
        summary = pd.DataFrame(results)
        summary.to_csv(stage / 'heldout_summary.csv', index=False)
        pd.concat(predictions, ignore_index=True).to_csv(stage / 'heldout_predictions.csv', index=False)
        dump(stage / 'manifest.json', {'schema_version': 1, 'labels': LABELS, 'target_mapping': TARGETS,
            'model_count': len(results), 'models': results,
            'note': 'Selected-window LOPO diagnostic; not continuous stream or independent final test.'})
        if a.out_dir.exists(): raise FileExistsError(a.out_dir)
        os.rename(stage, a.out_dir)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    print('\nCamera17 selected-window diagnostic:')
    for s in seconds_list:
        d = summary[summary.window_seconds == s]
        correct, total = d.camera17_squat_correct.sum(), d.camera17_squat_windows.sum()
        print(f'{s}s: {correct}/{total} ({correct/total:.2%}), models={len(d)}')
    print(f'Completed {len(results)} held-out models; saved: {a.out_dir}')


if __name__ == '__main__':
    main()
