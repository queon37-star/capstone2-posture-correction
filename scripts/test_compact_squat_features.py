#!/usr/bin/env python3
"""Focused LOPO study of knee-angle/leg-vertical asymmetry feature combinations.

Run from the capstone project root:
    python scripts/test_compact_squat_features.py

Inputs are the cached 263-feature data created by train_squat_extended.py --ablation.
All fitting (imputation, scaling, classifier) is performed within each train fold.
Results are exploratory; do not treat the best variant as independent test accuracy.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd

from audit_feature_usefulness import (
    ASYM_STATS, BASE_SIGNALS, BASE_STATS,
    calculate_predictions, expected_columns, validate_alignment,
)
import train_squat_baseline as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, default=Path('results/squat_feature_extension_ablation'))
    parser.add_argument('--out-dir', type=Path, default=Path('results/squat_compact_feature_test'))
    args = parser.parse_args()

    features = pd.read_csv(args.source_dir / 'extended_features.csv').replace([np.inf, -np.inf], np.nan)
    metadata = pd.read_csv(args.source_dir / 'used_repetitions.csv')
    validate_alignment(features, metadata, args.source_dir)
    base_cols = expected_columns(BASE_SIGNALS, BASE_STATS)
    groups = {name: [f'{name}_{st}' for st in ASYM_STATS]
              for name in ('asym_knee_angle', 'asym_leg_vertical', 'asym_hip_angle', 'asym_knee_motion')}

    # Keep order stable. Evaluate a small, hypothesis-driven shortlist, not a brute-force search.
    variants = OrderedDict([
        ('Baseline160', []),
        ('KneeAngle165', ['asym_knee_angle']),
        ('LegVertical165', ['asym_leg_vertical']),
        ('KneeLeg170', ['asym_knee_angle', 'asym_leg_vertical']),
        ('KneeLegHip175', ['asym_knee_angle', 'asym_leg_vertical', 'asym_hip_angle']),
        ('KneeLegMotion175', ['asym_knee_angle', 'asym_leg_vertical', 'asym_knee_motion']),
        ('PlusAsymmetry180', ['asym_knee_angle', 'asym_hip_angle', 'asym_knee_motion', 'asym_leg_vertical']),
    ])

    cols_by_variant = {name: base_cols + [c for g in signals for c in groups[g]]
                       for name, signals in variants.items()}
    for name, cols in cols_by_variant.items():
        if len(cols) != len(set(cols)):
            raise ValueError(f'{name}: duplicated columns')
        unknown = sorted(set(cols) - set(features.columns))
        if unknown:
            raise ValueError(f'{name}: missing features {unknown}')
        expected = 160 + 5*len(variants[name])
        assert len(cols) == expected, (name, len(cols), expected)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    y = metadata.label.to_numpy(dtype=int)
    template = base.build_models()['LogisticRegression']
    metrics, video_rows, person_rows = [], [], []
    prediction_table = metadata[['video_id', 'person_id', 'repetition_number', 'label']].copy()

    for name, cols in cols_by_variant.items():
        predictions, m, folds = calculate_predictions(features[cols], metadata, template)
        prediction_table['pred_' + name] = predictions
        metrics.append({'variant': name, 'features': len(cols),
                        'asymmetry_signals': '+'.join(variants[name]) if variants[name] else '(none)', **m})
        print(f'{name:18s} features={len(cols):3d} '
              f'BA={m["balanced_accuracy"]:.4f} incorrect_recall={m["incorrect_recall"]:.4f} '
              f'FN={m["false_negative_count"]:2d} FP={m["false_positive_count"]:2d}', flush=True)
        for v, group_idx in metadata.groupby('video_id').groups.items():
            idx = np.array(list(group_idx), dtype=int)
            yi, pi = y[idx], predictions[idx]
            video_rows.append({'variant': name, 'video_id': v,
                               'n_reps': len(idx), 'incorrect_count': int(np.sum(yi == 0)),
                               'FN': int(np.sum((yi == 0) & (pi == 1))),
                               'FP': int(np.sum((yi == 1) & (pi == 0)))})
        for f in folds:
            person_rows.append({'variant': name, **f})

    comparison = pd.DataFrame(metrics)
    comparison.to_csv(args.out_dir / 'comparison.csv', index=False)
    pd.DataFrame(video_rows).to_csv(args.out_dir / 'per_video_errors.csv', index=False)
    pd.DataFrame(person_rows).to_csv(args.out_dir / 'per_person_metrics.csv', index=False)
    prediction_table.to_csv(args.out_dir / 'lopo_predictions.csv', index=False)

    print('\n=== Compact combinations ===')
    print(comparison[['variant', 'features', 'accuracy', 'balanced_accuracy',
                      'incorrect_recall', 'incorrect_f1', 'false_negative_count',
                      'false_positive_count']].round(4).to_string(index=False))
    print('\n=== Video FN for compact two-group candidate and full 180 ===')
    pivot = pd.DataFrame(video_rows).pivot(index='video_id', columns='variant', values='FN')
    print(pivot[['Baseline160','KneeLeg170','PlusAsymmetry180']].to_string())
    print('\nSaved:', args.out_dir.resolve())
    print('Exploratory evaluation only: variants were selected after looking at these nine people.')


if __name__ == '__main__':
    main()
