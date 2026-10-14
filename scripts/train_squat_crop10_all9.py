#!/usr/bin/env python3
"""Fit the provisional PlusAsymmetry180 + 10% crop classifier on all nine people.

This is a demo/deployment artifact, NOT a held-out evaluation. For reported
performance use separately trained person-held-out fold artifacts/results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from compare_squat_lopo_augmentation import VIDEOS, trim_indices, extract_vector


def make_augmented_rows(samples, labels, feature_names, extractor=None, fraction=0.10):
    """Full, start-trim, end-trim, both-trim for each training repetition."""
    if extractor is None:
        from train_squat_extended import extract_feature_groups
        extractor = extract_feature_groups
    if len(samples) != len(labels) or not len(samples):
        raise ValueError('Expected nonempty samples and matching labels')
    X, y = [], []
    for sequence, label in zip(samples, labels):
        if int(label) not in (0, 1):
            raise ValueError('Labels must be 0=bad or 1=good')
        seq = np.asarray(sequence)
        if seq.ndim != 3 or seq.shape[1:] != (17, 3) or len(seq) < 15:
            raise ValueError(f'Expected [T,17,3] T>=15, got {seq.shape}')
        for start, end in trim_indices(len(seq), fraction):
            X.append(extract_vector(seq[start:end], feature_names, extractor))
            y.append(int(label))
    X = np.stack(X).astype(np.float64)
    y = np.asarray(y, dtype=int)
    if not np.isfinite(X).any(axis=1).all():
        raise ValueError('At least one training sequence contains no valid features')
    return X, y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'))
    p.add_argument('--template-model', type=Path, default=Path('models/squat_plus_asymmetry180.joblib'))
    p.add_argument('--output', type=Path, default=Path('models/squat_plus_asymmetry180_crop10_all9.joblib'))
    args = p.parse_args()
    source = joblib.load(args.template_model)  # ONLY self-trained, trusted artifacts.
    if source.get('feature_variant') != 'PlusAsymmetry180':
        raise ValueError('Template must use PlusAsymmetry180 feature schema')
    names = source.get('feature_names')
    if not isinstance(names, list) or len(names) != 180 or len(set(names)) != 180:
        raise ValueError('Template must contain exactly 180 unique features')

    index = pd.read_csv(args.data_dir / 'index.csv')
    required = {'file', 'video_id', 'person_id', 'repetition_number', 'label'}
    if not required.issubset(index.columns):
        raise ValueError(f'index.csv missing {sorted(required-set(index.columns))}')
    if len(index) != 195 or set(index.video_id) != set(VIDEOS) or index.person_id.nunique() != 9:
        raise ValueError('Expected exactly 195 repetitions, nine videos and nine subjects')
    if index[list(required)].isna().any().any():
        raise ValueError('Missing metadata in index.csv')
    if index.duplicated(['video_id', 'repetition_number']).any():
        raise ValueError('Duplicate video/repetition pair')
    if ((index.groupby('video_id').person_id.nunique()!=1).any() or
        (index.groupby('person_id').video_id.nunique()!=1).any()):
        raise ValueError('Expected a one-to-one person-to-video relationship')
    if set(index.label.astype(int)) != {0, 1}:
        raise ValueError('Expected binary 0/1 labels')

    sequences, labels = [], []
    for row in index.itertuples(index=False):
        path = args.data_dir / str(row.file)
        with np.load(path, allow_pickle=False) as pack:
            seq = np.asarray(pack['keypoints'], dtype=np.float32)
            saved_label = int(np.asarray(pack['label']).item())
            saved_person = int(np.asarray(pack['person_id']).item())
        if saved_label != int(row.label) or saved_person != int(row.person_id):
            raise ValueError(f'Index/NPZ mismatch: {path}')
        sequences.append(seq)
        labels.append(saved_label)

    X, y = make_augmented_rows(sequences, labels, names)
    expected = 4 * len(index)
    if X.shape != (expected, 180) or len(y) != expected:
        raise ValueError(f'Expected {(expected,180)} rows, found {X.shape}')
    print('Full-data prototype training: 195 repetitions x 4 variants =', len(y), 'rows')
    print('Train participants:', sorted(index.person_id.unique().tolist()))
    print('Train class counts:', {int(c):int((y==c).sum()) for c in (0,1)})
    model = clone(source['pipeline']).fit(X, y)
    classes = set(int(i) for i in model.classes_)
    if classes != {0, 1}:
        raise ValueError(f'Unexpected fitted classes: {classes}')
    artifact = {
        'pipeline': model,
        # This field denotes the FEATURE SCHEMA expected by run_squat_video_stream_tunable.py.
        'feature_variant': 'PlusAsymmetry180',
        'training_variant': 'PlusAsymmetry180_crop10_all9',
        'feature_names': names,
        'feature_groups': ['base', 'asymmetry'],
        'crop_fraction': 0.10,
        'training_repetitions': int(len(index)),
        'training_feature_rows': int(len(y)),
        'training_video_ids': sorted(index.video_id.unique().tolist()),
        'training_person_ids': sorted(int(i) for i in index.person_id.unique().tolist()),
        'label_map': {0: 'incorrect', 1: 'correct'},
        'extractor_module': 'train_squat_extended.py',
        'source': 'REHAB24-6 tracked YOLOv8m Pose camera17, manual repetition segments',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'sklearn_version': sklearn.__version__,
        'warning': 'In-sample full-data prototype; performance must come from separate LOPO models or new people.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    loaded = joblib.load(args.output)
    assert loaded['training_feature_rows'] == 780
    assert len(loaded['feature_names']) == 180
    print('Saved prototype artifact:', args.output)
    print('NOT an LOPO estimate. This model contains ALL NINE dataset participants.')


if __name__ == '__main__':
    main()
