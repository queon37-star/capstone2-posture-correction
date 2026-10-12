#!/usr/bin/env python3
"""Exploratory leave-one-person-out feature usefulness audit.

Consumes cached features produced by train_squat_extended.py --ablation.
No pose extraction, video decoding, or retraining of neural networks.

Run from project root with train_squat_baseline.py present in scripts/:
  python scripts/audit_feature_usefulness.py
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

import train_squat_baseline as base

DEFAULT_SOURCE = Path('results/squat_feature_extension_ablation')
DEFAULT_OUTPUT = Path('results/squat_feature_usefulness')

# Names match train_squat_baseline.extract_features() exactly.
BASE_SIGNALS = OrderedDict([
    ('left_knee_angle', 'Left projected knee angle'),
    ('right_knee_angle', 'Right projected knee angle'),
    ('knee_angle_mean', 'Average projected knee angle'),
    ('left_hip_angle', 'Left projected hip angle'),
    ('right_hip_angle', 'Right projected hip angle'),
    ('torso_tilt', 'Projected torso tilt from image vertical'),
    ('hip_ankle_vertical', 'Hip-to-ankle vertical image distance / torso length'),
    ('hip_knee_vertical_left', 'Left knee-to-hip vertical image distance / torso length'),
    ('hip_knee_vertical_right', 'Right knee-to-hip vertical image distance / torso length'),
    ('knee_ankle_horizontal_left', 'Left knee-to-ankle image x offset / torso length'),
    ('knee_ankle_horizontal_right', 'Right knee-to-ankle image x offset / torso length'),
    ('knee_separation', 'Left-right knee image x separation / torso length'),
    ('ankle_separation', 'Left-right ankle image x separation / torso length'),
    ('knee_minus_ankle_separation', 'Difference between knee and ankle image x separations'),
    ('hip_relative_to_ankles_x', 'Hip vs ankle midpoint image x offset / torso length'),
    ('hip_y_from_start', 'Hip vertical movement relative to first 15 percent of rep'),
])
ASYM_SIGNALS = OrderedDict([
    ('asym_knee_angle', 'Absolute left-right knee angle difference'),
    ('asym_hip_angle', 'Absolute left-right hip angle difference'),
    ('asym_knee_motion', 'Difference between successive left/right knee angle changes'),
    ('asym_leg_vertical', 'Absolute left-right knee-to-hip image vertical distance difference'),
])
BASE_STATS = ('mean', 'std', 'min', 'max', 'q10', 'median', 'q90', 'range', 'first', 'last')
ASYM_STATS = ('mean', 'std', 'max', 'q90', 'range')


def expected_columns(signal_names, stat_names):
    return [f'{signal}_{stat}' for signal in signal_names for stat in stat_names]


def calculate_predictions(feature_df, metadata, template):
    # Foldwise imputation and z-scaling are contained in the original Pipeline.
    x = feature_df.to_numpy(dtype=float)
    y = metadata.label.to_numpy(dtype=int)
    groups = metadata.person_id.to_numpy(dtype=int)
    preds = np.full(len(y), -1, dtype=int)
    subject_metrics = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        assert not (set(groups[train]) & set(groups[test])), 'Person split leaked'
        clf = clone(template)
        clf.fit(x[train], y[train])
        preds[test] = clf.predict(x[test])
        m = base.score(y[test], preds[test])
        m['person_id'] = int(groups[test[0]])
        m['test_repetitions'] = int(len(test))
        m['test_incorrect'] = int((y[test] == 0).sum())
        subject_metrics.append(m)
    assert np.all(preds != -1)
    metrics = base.score(y, preds)
    metrics['fold_mean_balanced_accuracy'] = float(np.mean([
        r['balanced_accuracy'] for r in subject_metrics]))
    metrics['false_negative_count'] = int(np.sum((y == 0) & (preds == 1)))
    metrics['false_positive_count'] = int(np.sum((y == 1) & (preds == 0)))
    metrics['error_count'] = int(np.sum(y != preds))
    return preds, metrics, subject_metrics


def make_variants(base_cols, asym_cols):
    full = base_cols + asym_cols
    variants = OrderedDict()
    variants['Baseline160'] = ('reference', 'Baseline160', base_cols, 'baseline')
    variants['PlusAsymmetry180'] = ('reference', 'PlusAsymmetry180', full, 'baseline_plus_asym')

    for signal in ASYM_SIGNALS:
        add = [f'{signal}_{stat}' for stat in ASYM_STATS]
        variants['AddOnly_' + signal] = ('add_asym_signal', 'Baseline160', base_cols + add, signal)

    for signal, stats in [(s, BASE_STATS) for s in BASE_SIGNALS] + [(s, ASYM_STATS) for s in ASYM_SIGNALS]:
        removal = set(expected_columns([signal], stats))
        variants['Without_' + signal] = (
            'remove_signal', 'PlusAsymmetry180', [c for c in full if c not in removal], signal)

    for stat in BASE_STATS:
        removal = {f'{s}_{stat}' for s in BASE_SIGNALS}
        variants['WithoutBaseStat_' + stat] = (
            'remove_base_stat', 'PlusAsymmetry180', [c for c in full if c not in removal], stat)

    for stat in ASYM_STATS:
        removal = {f'{s}_{stat}' for s in ASYM_SIGNALS}
        variants['WithoutAsymStat_' + stat] = (
            'remove_asym_stat', 'PlusAsymmetry180', [c for c in full if c not in removal], stat)
    return variants


def diff_if_useful(score, reference, analysis_type):
    # Positive result means keeping/adding the feature helped (exploratory only).
    if analysis_type == 'add_asym_signal':
        return (score - reference) * 100
    return (reference - score) * 100


def validate_alignment(features, metadata, source):
    key_cols = ['video_id', 'person_id', 'repetition_number', 'label', 'file']
    missing = sorted(set(key_cols) - set(metadata.columns))
    if missing:
        raise ValueError(f'used_repetitions.csv missing: {missing}')
    if len(features) != len(metadata):
        raise ValueError(f'Rows differ: {len(features)} features, {len(metadata)} metadata')
    if metadata.duplicated(['video_id', 'repetition_number']).any():
        raise ValueError('Duplicate video/repetition IDs in metadata')
    if features.columns.duplicated().any():
        raise ValueError('Duplicate feature column names')
    if metadata.label.isna().any() or not set(metadata.label.astype(int).unique()) == {0, 1}:
        raise ValueError('Both correct (1) and incorrect (0) labels required')
    if metadata.person_id.nunique() < 3:
        raise ValueError('At least 3 subjects are needed')
    # Check metadata against the tracked index, without assuming any row order there.
    source_index = Path('data/processed/REHAB24-6/camera17_yolov8m_tracked/index.csv')
    if source_index.exists():
        index = pd.read_csv(source_index)
        left = metadata[key_cols].sort_values(key_cols).reset_index(drop=True)
        right = index[key_cols].sort_values(key_cols).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError('Cached feature metadata does not match tracked index; re-run --ablation first')
    print(f'Input: {source}\nSamples: {len(metadata)}, People: {metadata.person_id.nunique()}, classes: {metadata.label.value_counts().to_dict()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--quick', action='store_true', help='Run 2 references + 4 isolated asymmetry additions only')
    args = parser.parse_args()

    feature_path = args.source_dir / 'extended_features.csv'
    meta_path = args.source_dir / 'used_repetitions.csv'
    if not feature_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            'Cached feature files not found. First run:\n'
            'python scripts/train_squat_extended.py --ablation '
            '--out-dir results/squat_feature_extension_ablation')
    features = pd.read_csv(feature_path).replace([np.inf, -np.inf], np.nan)
    metadata = pd.read_csv(meta_path)
    validate_alignment(features, metadata, args.source_dir)

    base_cols = expected_columns(BASE_SIGNALS, BASE_STATS)
    asym_cols = expected_columns(ASYM_SIGNALS, ASYM_STATS)
    if len(base_cols) != 160 or len(asym_cols) != 20:
        raise AssertionError('Feature schema mismatch')
    required = base_cols + asym_cols
    missing = [c for c in required if c not in features.columns]
    if missing:
        raise ValueError(f'Missing expected features: {missing[:12]} (total {len(missing)})')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    catalog = []
    for group_name, signals, stats in [
        ('base', BASE_SIGNALS, BASE_STATS), ('asymmetry', ASYM_SIGNALS, ASYM_STATS)]:
        for s, description in signals.items():
            for st in stats:
                c = f'{s}_{st}'
                catalog.append({
                    'feature': c, 'group': group_name, 'signal': s,
                    'statistic': st, 'meaning': description,
                    'observed_fraction': float(features[c].notna().mean()),
                })
    pd.DataFrame(catalog).to_csv(args.out_dir / 'feature_catalog.csv', index=False)

    variants = make_variants(base_cols, asym_cols)
    if args.quick:
        variants = OrderedDict((k, v) for k, v in variants.items()
                               if v[0] in ('reference', 'add_asym_signal'))
    y = metadata.label.to_numpy(dtype=int)
    template = base.build_models()['LogisticRegression']

    scores, predictions, folds = {}, {}, {}
    rows = []
    for i, (variant, (kind, reference, columns, affected)) in enumerate(variants.items(), 1):
        pr, score, person_metrics = calculate_predictions(features[columns], metadata, template)
        scores[variant], predictions[variant], folds[variant] = score, pr, person_metrics
        rows.append({
            'variant': variant, 'test_type': kind,
            'reference': reference, 'signal_or_stat': affected,
            'features': len(columns), **score,
        })
        print(f'[{i:02d}/{len(variants)}] {variant:<36} n={len(columns):3d} '
              f'BA={score["balanced_accuracy"]:.4f} '
              f'FNrecall={score["incorrect_recall"]:.4f} '
              f'FN={score["false_negative_count"]:2d} '
              f'FP={score["false_positive_count"]:2d}', flush=True)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.out_dir / 'all_variants.csv', index=False)

    effects, person_effects = [], []
    for variant, (kind, reference, columns, affected) in variants.items():
        if kind == 'reference':
            continue
        sc, ref = scores[variant], scores[reference]
        item = {
            'variant': variant, 'test_type': kind,
            'reference': reference, 'signal_or_stat': affected,
            'n_features': len(columns),
            'usefulness_balanced_accuracy_pp': diff_if_useful(sc['balanced_accuracy'], ref['balanced_accuracy'], kind),
            'usefulness_incorrect_recall_pp': diff_if_useful(sc['incorrect_recall'], ref['incorrect_recall'], kind),
            'usefulness_incorrect_f1_pp': diff_if_useful(sc['incorrect_f1'], ref['incorrect_f1'], kind),
            'usefulness_accuracy_pp': diff_if_useful(sc['accuracy'], ref['accuracy'], kind),
            'FN_candidate': sc['false_negative_count'],
            'FP_candidate': sc['false_positive_count'],
            'FN_reference': ref['false_negative_count'],
            'FP_reference': ref['false_positive_count'],
        }
        change = []
        for person in sorted(metadata.person_id.unique()):
            fm = next(row for row in folds[variant] if row['person_id'] == person)
            rf = next(row for row in folds[reference] if row['person_id'] == person)
            effect = diff_if_useful(fm['balanced_accuracy'], rf['balanced_accuracy'], kind)
            change.append(effect)
            if kind in ('add_asym_signal', 'remove_signal'):
                person_effects.append({
                    'variant': variant, 'test_type': kind,
                    'person_id': person, 'signal': affected,
                    'usefulness_balanced_accuracy_pp': effect,
                    'usefulness_incorrect_recall_pp': diff_if_useful(
                        fm['incorrect_recall'], rf['incorrect_recall'], kind),
                })
        item.update({
            'people_helped': int(np.sum(np.array(change) > 1e-9)),
            'people_hurt': int(np.sum(np.array(change) < -1e-9)),
            'people_unchanged': int(np.sum(np.abs(change) <= 1e-9)),
        })
        effects.append(item)
    effect_df = pd.DataFrame(effects)
    effect_df.to_csv(args.out_dir / 'feature_group_effects.csv', index=False)
    pd.DataFrame(person_effects).to_csv(args.out_dir / 'person_effects.csv', index=False)

    prediction_table = metadata[['video_id', 'person_id', 'repetition_number', 'label']].copy()
    for v in ['Baseline160', 'PlusAsymmetry180'] + [f'AddOnly_{s}' for s in ASYM_SIGNALS]:
        prediction_table['pred_' + v] = predictions[v]
    prediction_table.to_csv(args.out_dir / 'key_predictions.csv', index=False)

    # Only report *observed*, held-out-person prediction changes, not causal error diagnosis.
    reference_preds = predictions['Baseline160']
    change_rows = []
    for s in ASYM_SIGNALS:
        v = f'AddOnly_{s}'
        pred = predictions[v]
        changed = np.flatnonzero(pred != reference_preds)
        for idx in changed:
            fixed = pred[idx] == y[idx]
            row = metadata.iloc[idx]
            change_rows.append({
                'variant': v,
                'video_id': row.video_id,
                'person_id': int(row.person_id),
                'repetition_number': int(row.repetition_number),
                'label': int(y[idx]),
                'pred_baseline': int(reference_preds[idx]),
                'pred_variant': int(pred[idx]),
                'outcome': 'fixed' if fixed else 'broken',
            })
    pd.DataFrame(change_rows, columns=[
        'variant', 'video_id', 'person_id', 'repetition_number', 'label',
        'pred_baseline', 'pred_variant', 'outcome']).to_csv(
            args.out_dir / 'individual_asymmetry_prediction_changes.csv', index=False)

    print('\n=== REFERENCE ===')
    print(comparison[comparison.test_type == 'reference'][
        ['variant', 'features', 'accuracy', 'balanced_accuracy', 'incorrect_recall', 'incorrect_f1']
    ].round(4).to_string(index=False))
    print('\n=== INDIVIDUAL ASYMMETRY ADDITIONS (positive = helped) ===')
    print(effect_df[effect_df.test_type == 'add_asym_signal'][
        ['signal_or_stat', 'usefulness_balanced_accuracy_pp', 'usefulness_incorrect_recall_pp',
         'people_helped', 'people_hurt']
    ].round(2).to_string(index=False))
    if not args.quick:
        print('\n=== LEAVE-ONE-SIGNAL-OUT: PLUS180 (positive = keeping signal helped) ===')
        print(effect_df[effect_df.test_type == 'remove_signal'][
            ['signal_or_stat', 'usefulness_balanced_accuracy_pp', 'usefulness_incorrect_recall_pp',
             'people_helped', 'people_hurt']
        ].round(2).sort_values('usefulness_balanced_accuracy_pp', ascending=False).to_string(index=False))
        print('\n=== REMOVE ONE STATISTIC FAMILY: PLUS180 (positive = keeping statistic helped) ===')
        print(effect_df[effect_df.test_type.str.contains('remove_.*stat', regex=True)][
            ['test_type', 'signal_or_stat', 'usefulness_balanced_accuracy_pp', 'usefulness_incorrect_recall_pp']
        ].round(2).to_string(index=False))
    print('\nSaved to:', args.out_dir.resolve())
    print('WARNING: exploratory results on the same nine people; no independent test.')
    print('Correlated feature groups can substitute for one another; removal impact is not a causal importance score.')


if __name__ == '__main__':
    main()
