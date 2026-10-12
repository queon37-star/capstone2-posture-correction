"""Refit the exploratory 180-feature candidate on all 9 subjects.

The published LOPO metrics come from train_squat_extended.py --ablation;
this full-data fit is solely for prototype inference.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
import train_squat_baseline as base
from train_squat_extended import extract_feature_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path,
                    default=Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'))
    ap.add_argument('--output', type=Path,
                    default=Path('models/squat_plus_asymmetry180.joblib'))
    args = ap.parse_args()
    df = pd.read_csv(args.data_dir / 'index.csv')
    if len(df) != 195 or df.person_id.nunique() != 9:
        raise RuntimeError('Dataset changed; re-evaluate before proceeding')
    rows = []
    for _, row in df.iterrows():
        with np.load(args.data_dir / str(row.file), allow_pickle=False) as z:
            groups = extract_feature_groups(z['keypoints'])
        feat = {}
        feat.update(groups['base'])
        feat.update(groups['asymmetry'])
        rows.append(feat)
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    assert X.shape == (195, 180), X.shape
    assert set(df.label.unique()) == {0, 1}
    model = clone(base.build_models()['LogisticRegression'])
    model.fit(X.to_numpy(dtype=float), df.label.to_numpy(dtype=int))
    artifact = {
        'pipeline': model,
        'feature_names': X.columns.tolist(),
        'feature_variant': 'PlusAsymmetry180',
        'label_map': {0: 'incorrect', 1: 'correct'},
        'training_source': 'REHAB24-6 camera17, 9 subjects, 195 reps',
        'validation_note': 'Exploratory LOPO BA=.8583, not independent external validation',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print('Saved:', args.output)
    print('Features:', X.shape[1], 'Subjects:', df.person_id.nunique())
    print('Prediction label 0=incorrect, 1=correct. Model is REFIT ON ALL DATA.')


if __name__ == '__main__':
    main()
