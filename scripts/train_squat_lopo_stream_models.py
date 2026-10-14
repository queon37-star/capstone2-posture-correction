"""Fit one PlusAsymmetry180 model per held-out REHAB24-6 participant.

Training features are extracted from manually segmented ground-truth training
repetitions of OTHER participants only. No features from the held-out person
enter the model fit. Detection uses the unchanged streaming state machine and
never reads annotations.

Caveat: 180-feature selection and detector hyperparameters were previously
explored on all nine participants. This is a held-out-weight audit, not a
truly untouched external generalization benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone


def validate_metadata(index: pd.DataFrame) -> None:
    required = {'file', 'person_id', 'video_id', 'label', 'repetition_number'}
    if not required.issubset(index.columns):
        raise ValueError(f'index.csv lacks fields {sorted(required - set(index.columns))}')
    if index.empty or index.isna()[list(required)].any().any():
        raise ValueError('Empty/partially missing index metadata')
    if index.duplicated(['video_id', 'repetition_number']).any():
        raise ValueError('Duplicate video_id/repetition_number')
    if not set(index.label.astype(int).unique()) <= {0, 1}:
        raise ValueError('Expected binary correctness labels (0 bad, 1 good)')
    # Current corpus has one video per person, so each held-out video also
    # determines exactly which person's repetitions must be excluded.
    if (index.groupby('video_id').person_id.nunique() != 1).any():
        raise ValueError('Every video must contain exactly one subject')
    if (index.groupby('person_id').video_id.nunique() != 1).any():
        raise ValueError('One video per subject is expected for this experiment')


def fit_heldout_models(index: pd.DataFrame, x: np.ndarray,
                       feature_names: list[str], template, output_dir: Path):
    """Clones an UNFITTED estimator for each fold. Does not reuse learned weights."""
    validate_metadata(index)
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (len(index), len(feature_names)):
        raise ValueError(f'Expected X={(len(index), len(feature_names))}, got {x.shape}')
    if np.any(~np.isfinite(x).any(axis=1)):
        raise ValueError('One or more samples lack any finite feature')
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for video in sorted(index.video_id.unique()):
        heldout = index.loc[index.video_id == video, 'person_id'].iloc[0]
        train_mask = (index.person_id != heldout).to_numpy()
        test_mask = ~train_mask
        train_people = sorted(index.loc[train_mask, 'person_id'].astype(int).unique().tolist())
        assert int(heldout) not in train_people
        y_train = index.loc[train_mask, 'label'].astype(int).to_numpy()
        if set(np.unique(y_train)) != {0, 1}:
            raise ValueError(f'Fold {video}: training lacks a class')
        # sklearn.base.clone copies ONLY estimator params, not learned coef_,
        # scaler statistics, or imputer statistics from full-data template.
        model = clone(template)
        model.fit(x[train_mask], y_train)
        record = {
            'pipeline': model,
            'feature_names': list(feature_names),
            'feature_groups': ['base', 'asymmetry'],
            'feature_variant': 'PlusAsymmetry180',
            'label_map': {0: 'incorrect', 1: 'correct'},
            'held_out_video': str(video),
            'held_out_person_id': int(heldout),
            'training_person_ids': train_people,
            'training_sample_count': int(train_mask.sum()),
            'held_out_sample_count': int(test_mask.sum()),
            'evaluation_note': 'Held-out classifier weights only: detection thresholds and feature selection were developed on the nine-person corpus.',
        }
        filename = output_dir / f'{video}.joblib'
        joblib.dump(record, filename)
        summary.append({
            'held_out_video': str(video),
            'held_out_person_id': int(heldout),
            'training_people': len(train_people),
            'training_repetitions': int(train_mask.sum()),
            'held_out_repetitions': int(test_mask.sum()),
            'model_path': str(filename),
        })
        print(f'{video}: heldout_person={heldout}, train={train_mask.sum()} reps '
              f'({len(train_people)} subjects), test={test_mask.sum()} reps -> {filename}', flush=True)
    (output_dir / 'manifest.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'))
    parser.add_argument('--base-model', type=Path,
                        default=Path('models/squat_plus_asymmetry180.joblib'))
    parser.add_argument('--out-dir', type=Path, default=Path('models/squat_lopo_stream'))
    args = parser.parse_args()

    if not args.data_dir.joinpath('index.csv').is_file():
        raise FileNotFoundError(f'Missing {args.data_dir}/index.csv')
    index = pd.read_csv(args.data_dir / 'index.csv')
    validate_metadata(index)
    # Check the dataset identity so accidentally pointing to another
    # differently indexed corpus never silently changes the evaluation.
    if index.video_id.nunique() != 9 or index.person_id.nunique() != 9:
        raise ValueError('Expected nine REHAB24-6 participants and nine videos')
    artifact = joblib.load(args.base_model)  # Load only your own trusted artifact.
    if artifact.get('feature_variant') != 'PlusAsymmetry180':
        raise ValueError('Expected self-trained PlusAsymmetry180 model')
    feature_names = list(artifact['feature_names'])
    if len(feature_names) != 180 or len(set(feature_names)) != 180:
        raise ValueError('Unexpected 180-feature schema')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_squat_extended import extract_feature_groups

    rows = []
    for r in index.itertuples(index=False):
        sample = args.data_dir / str(r.file)
        with np.load(sample, allow_pickle=False) as npz:
            seq = npz['keypoints']
            label = int(np.asarray(npz['label']).item())
            person = int(np.asarray(npz['person_id']).item())
        if label != int(r.label) or person != int(r.person_id):
            raise ValueError(f'Index and NPZ mismatch for {sample}')
        group = extract_feature_groups(seq)
        features = dict(group['base'])
        features.update(group['asymmetry'])
        if set(features) != set(feature_names):
            raise ValueError(f'Feature schema mismatch for {sample}')
        rows.append([features[name] for name in feature_names])
    x = np.asarray(rows, dtype=np.float64)
    print(f'Features: {len(index)} examples x {len(feature_names)} features; '
          f'valid={np.isfinite(x).mean():.3f} fraction', flush=True)
    fit_heldout_models(index, x, feature_names, artifact['pipeline'], args.out_dir)
    print('FINISHED: trained nine leave-one-person-out classifier models.')
    print('No model was fitted on its corresponding held-out person.')
    print('IMPORTANT: preselected features and detection thresholds were NOT held out.')


if __name__ == '__main__':
    main()
