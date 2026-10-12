#!/usr/bin/env python3
"""Compare unchanged 160-feature squat LR baseline against geometry/phase extensions.

Run from ~/capstone2/pose_training after copying beside train_squat_baseline.py:
    python scripts/train_squat_extended.py
    python scripts/train_squat_extended.py --ablation

Input: camera17_yolov8m_tracked/index.csv plus YOLO17 .npz files.
Labels 0=incorrect, 1=correct. No video/person identifiers used as features.
No additional neural-network inference. Phase detection uses 2D hip trajectory.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

# Deliberately import original module so the 160 original features, model and
# scoring definitions are EXACTLY those used for the published baseline run.
import train_squat_baseline as base

DEFAULT_IN = Path('data/processed/REHAB24-6/camera17_yolov8m_tracked')
DEFAULT_OUT = Path('results/squat_feature_extension')
LS, RS, LH, RH, LK, RK, LA, RA = base.LS, base.RS, base.LH, base.RH, base.LK, base.RK, base.LA, base.RA
PHASES = ('descent', 'bottom', 'ascent')
EVENTS = ('descent25', 'descent75', 'bottom', 'ascent25', 'ascent75')


def safe_mean(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def safe_stat(a, what):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan
    if what == 'mean': return float(a.mean())
    if what == 'std': return float(a.std())
    if what == 'max': return float(a.max())
    if what == 'q90': return float(np.quantile(a, 0.9))
    if what == 'range': return float(a.max() - a.min())
    raise ValueError(what)


def stats5(name, values):
    return {f'{name}_{s}': safe_stat(values, s)
            for s in ('mean', 'std', 'max', 'q90', 'range')}


def finite_interp(x):
    """Fill missing hip observations for *event localization only*.

    Raw per-frame pose features remain NaN at missing joints. We do not make up
    observations from interpolated poses.
    """
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if good.sum() < 3:
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def running_median(x, width=7):
    # Centered smoothing prevents one noisy hip keypoint creating a false bottom.
    return pd.Series(x).rolling(width, center=True, min_periods=1).median().to_numpy()


def get_phase_indices(t, bottom_idx):
    """Approximate descent, deepest moment, ascent by hip trajectory.

    Phase features are only extracted if there are sufficient samples on both
    sides of the bottom. This avoids a false interpretation for incomplete reps.
    """
    phases = {k: np.array([], dtype=int) for k in PHASES}
    if bottom_idx is None or bottom_idx < max(4, int(.12*t)) or bottom_idx > min(t-5, int(.88*t)):
        return phases
    # Descent/ascent: central 60% of each segment; bottom: +/- 5% of repetition.
    d0 = int(np.floor(.2 * bottom_idx))
    d1 = int(np.ceil(.8 * bottom_idx)) + 1
    rest = (t - 1) - bottom_idx
    a0 = bottom_idx + int(np.floor(.2 * rest))
    a1 = bottom_idx + int(np.ceil(.8 * rest)) + 1
    radius = max(2, int(round(t * .05)))
    phases['descent'] = np.arange(d0, min(d1, t))
    phases['bottom'] = np.arange(max(0, bottom_idx - radius), min(t, bottom_idx + radius + 1))
    phases['ascent'] = np.arange(a0, min(a1, t))
    return phases


def near_value(a, idx, radius=2):
    i = int(round(idx))
    a = np.asarray(a, dtype=float)
    return safe_mean(a[max(0, i-radius): min(len(a), i+radius+1)])


def extract_feature_groups(seq):
    """Return exact original 160 features and new interpretable feature groups.

    Keypoint layout: [T, 17, 3] (x/y pixels + detection confidence).
    All 2D geometry is scale normalized by median torso length where needed.
    Knee-in features are masked when ankle separation is too small for reliable
    front-view geometry, reducing spurious side-view ratio features.
    """
    original = base.extract_features(seq)  # Validate shape and scale, exactly as before.
    xy = np.asarray(seq[:, :, :2], dtype=float).copy()
    cf = np.asarray(seq[:, :, 2], dtype=float)
    xy[(cf < 0.30) | ~np.isfinite(cf)] = np.nan
    t = len(seq)

    shoulders = base.mid(xy[:, LS], xy[:, RS])
    hips = base.mid(xy[:, LH], xy[:, RH])
    ankles = base.mid(xy[:, LA], xy[:, RA])
    torso = base.distance(shoulders, hips)
    good_scale = torso[np.isfinite(torso) & (torso > 10)]
    scale = float(np.median(good_scale))
    assert np.isfinite(scale) and scale > 0

    l_knee = base.angle(xy[:, LH], xy[:, LK], xy[:, LA])
    r_knee = base.angle(xy[:, RH], xy[:, RK], xy[:, RA])
    l_hip = base.angle(xy[:, LS], xy[:, LH], xy[:, LK])
    r_hip = base.angle(xy[:, RS], xy[:, RH], xy[:, RK])
    knee_mean = base.mid_scalar(l_knee, r_knee)
    hip_mean = base.mid_scalar(l_hip, r_hip)
    tilt = np.degrees(np.arctan2(np.abs(shoulders[:, 0]-hips[:, 0]),
                                 np.abs(shoulders[:, 1]-hips[:, 1])))

    knee_diff = np.abs(l_knee-r_knee)
    hip_diff = np.abs(l_hip-r_hip)
    # Per-frame left/right difference in change of knee angle.
    # Unlike raw speed, this is insensitive to moving both legs in synchrony.
    angle_step_mismatch = np.full(t, np.nan)
    angle_step_mismatch[1:] = np.abs(np.diff(l_knee) - np.diff(r_knee))
    left_depth = (xy[:, LK, 1] - xy[:, LH, 1]) / scale
    right_depth = (xy[:, RK, 1] - xy[:, RH, 1]) / scale
    leg_vertical_asym = np.abs(left_depth - right_depth)

    asymmetry = {}
    for name, vals in {
        'asym_knee_angle': knee_diff,
        'asym_hip_angle': hip_diff,
        'asym_knee_motion': angle_step_mismatch,
        'asym_leg_vertical': leg_vertical_asym,
    }.items():
        asymmetry.update(stats5(name, vals))

    # Projected frontal alignment. Never force ratios when the ankles overlap
    # in half-profile view. A missing ratio is imputed within the training fold.
    ankle_sep = np.abs(xy[:, LA, 0]-xy[:, RA, 0])
    knee_sep = np.abs(xy[:, LK, 0]-xy[:, RK, 0])
    width_ok = np.isfinite(ankle_sep) & (ankle_sep >= .35 * scale)
    width_ok &= np.isfinite(knee_sep)
    with np.errstate(divide='ignore', invalid='ignore'):
        width_ratio = np.where(width_ok, np.clip(knee_sep / ankle_sep, 0, 4), np.nan)
    # Positive means knees project closer to center than the ankles.
    ankle_mid_x = (xy[:, LA, 0] + xy[:, RA, 0]) / 2.
    l_inward = (np.abs(xy[:, LA, 0] - ankle_mid_x) - np.abs(xy[:, LK, 0] - ankle_mid_x))/scale
    r_inward = (np.abs(xy[:, RA, 0] - ankle_mid_x) - np.abs(xy[:, RK, 0] - ankle_mid_x))/scale
    l_inward[~width_ok] = np.nan
    r_inward[~width_ok] = np.nan
    knee_width_loss = np.where(width_ok, (ankle_sep - knee_sep)/scale, np.nan)
    alignment = {}
    for name, vals in {
        'align_knee_to_ankle_width_ratio': width_ratio,
        'align_knee_width_loss': knee_width_loss,
        'align_left_inward': l_inward,
        'align_right_inward': r_inward,
    }.items():
        alignment.update(stats5(name, vals))
    alignment['align_width_usable_fraction'] = float(width_ok.mean())

    # Larger hip y means deeper squat in normal downward-positive image coords.
    ref_start = safe_mean(hips[:max(1, int(np.ceil(.15 * t))), 1])
    depth = (hips[:, 1] - ref_start)/scale if np.isfinite(ref_start) else np.full(t, np.nan)
    hip_y_interp = finite_interp(hips[:, 1])
    bottom = int(np.argmax(running_median(hip_y_interp))) if hip_y_interp is not None else None
    ph_idx = get_phase_indices(t, bottom)

    signals = {
        'knee_mean': knee_mean,
        'hip_mean': hip_mean,
        'knee_asym': knee_diff,
        'torso_tilt': tilt,
        'squat_depth': depth,
        'knee_width_loss': knee_width_loss,
        'knee_width_ratio': width_ratio,
    }
    phase_features = {}
    for name, vals in signals.items():
        for part in PHASES:
            phase_features[f'phase_{name}_{part}_mean'] = safe_mean(vals[ph_idx[part]])
    for name in ('knee_mean', 'torso_tilt', 'knee_asym'):
        if len(ph_idx['bottom']):
            phase_features[f'phase_{name}_bottom_minus_start'] = (
                safe_mean(signals[name][ph_idx['bottom']]) - safe_mean(signals[name][:max(1, int(.15*t))]))
        else:
            phase_features[f'phase_{name}_bottom_minus_start'] = np.nan

    # Stage-aligned landmarks, not raw time percentage, so descent/ascent
    # rates can differ across participants. Local averages suppress jitter.
    trajectory = {}
    for name, vals in signals.items():
        for phase_name in EVENTS:
            if bottom is None or len(ph_idx['descent']) == 0 or len(ph_idx['ascent']) == 0:
                value = np.nan
            else:
                rest = t - 1 - bottom
                at = {
                    'descent25': .25 * bottom,
                    'descent75': .75 * bottom,
                    'bottom': float(bottom),
                    'ascent25': bottom + .25 * rest,
                    'ascent75': bottom + .75 * rest,
                }[phase_name]
                value = near_value(vals, at, radius=2)
            trajectory[f'traj_{name}_{phase_name}'] = value

    # Quality cues are *not* anatomy. Keep separately identifiable so an
    # ablation can test if confidence/occlusion is causing shortcuts.
    bilateral_legs = np.isfinite(xy[:, [LH, RH, LK, RK, LA, RA], :]).all(axis=(1, 2))
    quality = {
        'quality_bilateral_leg_visibility': float(np.mean(bilateral_legs)),
        'quality_bilateral_knee_angle_visibility': float(np.mean(np.isfinite(l_knee) & np.isfinite(r_knee))),
        'quality_mean_leg_confidence': safe_mean(cf[:, [LH, RH, LK, RK, LA, RA]]),
    }
    return {
        'base': original,
        'asymmetry': asymmetry,
        'alignment': alignment,
        'phase': phase_features,
        'trajectory': trajectory,
        'quality': quality,
    }


def extract_full_features(seq):
    """Stable inference API: returns the complete extended feature dictionary."""
    groups = extract_feature_groups(seq)
    out = {}
    for group in ('base', 'asymmetry', 'alignment', 'phase', 'trajectory', 'quality'):
        out.update(groups[group])
    return out


def make_feature_matrix(group_rows, active_groups):
    records = []
    for groups in group_rows:
        merged = {}
        for name in active_groups:
            merged.update(groups[name])
        records.append(merged)
    result = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan)
    # Keep globally empty features to keep feature schema stable. The original
    # SimpleImputer(keep_empty_features=True) turns them into zeros per fold.
    if result.isna().all(axis=1).any():
        raise ValueError('A sample has no usable features')
    return result


def eval_variant(dataframe, features, label, template):
    x = features.to_numpy(dtype=float)
    y = dataframe.label.to_numpy(dtype=int)
    groups = dataframe.person_id.to_numpy(dtype=int)
    preds = np.full(len(y), -1, dtype=int)
    fold = []
    loo = LeaveOneGroupOut()
    for train, test in loo.split(x, y, groups):
        assert not (set(groups[train]) & set(groups[test]))
        pipe = clone(template)
        pipe.fit(x[train], y[train])
        preds[test] = pipe.predict(x[test])
        metrics = base.score(y[test], preds[test])
        metrics.update({
            'person_id': int(groups[test[0]]),
            'test_repetitions': int(len(test)),
            'test_incorrect': int((y[test] == 0).sum()),
        })
        fold.append(metrics)
    assert np.all(preds != -1)
    overall = base.score(y, preds)
    overall['fold_mean_balanced_accuracy'] = float(np.mean([f['balanced_accuracy'] for f in fold]))
    return preds, overall, pd.DataFrame(fold).sort_values('person_id')


def plot_confusion(y, pred, title, output):
    cm = confusion_matrix(y, pred, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=['Correct', 'Incorrect'])
    ax.set_yticks([0, 1], labels=['Correct', 'Incorrect'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground truth')
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha='center', va='center',
                color='white' if value > cm.max()/2 else 'black', fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return cm.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_IN)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--ablation', action='store_true', help='Evaluate incremental feature subsets (exploratory).')
    args = parser.parse_args()
    idx_file = args.data_dir / 'index.csv'
    if not idx_file.exists():
        raise FileNotFoundError(idx_file)
    df = pd.read_csv(idx_file)
    required = ['file', 'person_id', 'label', 'video_id', 'repetition_number']
    if any(x not in df.columns for x in required):
        raise ValueError(f'Missing index fields: {set(required) - set(df.columns)}')
    if df.duplicated(['video_id', 'repetition_number']).any():
        raise ValueError('Duplicate repetitions in index')
    print(f'Samples in index: {len(df)}; People: {df.person_id.nunique()}')

    feature_rows, valid_idx, bad = [], [], []
    for idx, r in df.iterrows():
        try:
            with np.load(args.data_dir / str(r['file']), allow_pickle=False) as npz:
                if int(npz['label']) != int(r.label) or int(npz['person_id']) != int(r.person_id):
                    raise ValueError('Index/NPZ label or person mismatch')
                row = extract_feature_groups(npz['keypoints'])
            feature_rows.append(row)
            valid_idx.append(idx)
        except Exception as ex:
            bad.append((str(r['file']), str(ex)))
    if bad:
        print('INVALID repetitions:')
        for name, why in bad:
            print(' ', name, why)
        warnings.warn(f'Excluded {len(bad)} invalid repetitions')
    df = df.loc[valid_idx].reset_index(drop=True)
    if len(df) < 30 or df.person_id.nunique() < 3 or set(df.label.unique()) != {0, 1}:
        raise ValueError('Insufficient valid labeled data for grouped evaluation')
    print('Valid samples:', len(df), 'Class distribution:', df.label.value_counts().to_dict())

    full_groups = ('base', 'asymmetry', 'alignment', 'phase', 'trajectory', 'quality')
    choices = {
        'Baseline160': ('base',),
        'Extended': full_groups,
    }
    if args.ablation:
        choices = {
            'Baseline160': ('base',),
            'PlusAsymmetry': ('base', 'asymmetry'),
            'PlusAlignment': ('base', 'asymmetry', 'alignment'),
            'PlusPhases': ('base', 'asymmetry', 'alignment', 'phase'),
            'PlusTrajectory': ('base', 'asymmetry', 'alignment', 'phase', 'trajectory'),
            'Extended': full_groups,
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    template = base.build_models()['LogisticRegression']
    y = df.label.to_numpy(dtype=int)
    prediction_table = df[['video_id', 'person_id', 'repetition_number', 'label']].copy()
    out_report = {
        'description': '2D pose squat feature ablation, exploratory nine-person LOPO',
        'data': 'REHAB24-6 Camera17 tracked YOLOv8m-pose',
        'label_map': {'0': 'incorrect', '1': 'correct'},
        'n_repetitions': int(len(df)),
        'n_people': int(df.person_id.nunique()),
        'class_counts': {str(k): int(np.sum(y == k)) for k in [0, 1]},
        'model': 'Original LogisticRegression pipeline (impute median, z-scale, balanced, C=.1)',
        'variants': {},
    }
    matrices = {}
    summary = []
    all_preds = {}
    per_person = {}
    for variant, groups in choices.items():
        feats = make_feature_matrix(feature_rows, groups)
        matrices[variant] = feats
        preds, metrics, folds = eval_variant(df, feats, variant, template)
        all_preds[variant] = preds
        per_person[variant] = folds
        folds.to_csv(args.out_dir / f'{variant}_per_person.csv', index=False)
        prediction_table['pred_' + variant] = preds
        cm = plot_confusion(y, preds, f'{variant} (LOPO)',
                            args.out_dir / f'confusion_{variant}.png')
        out_report['variants'][variant] = {
            'feature_groups': list(groups),
            'feature_count': len(feats.columns),
            'pooled_out_of_person': metrics,
            'per_person': folds.to_dict(orient='records'),
            'confusion_matrix_labels': ['correct', 'incorrect'],
            'confusion_matrix': cm,
        }
        summary.append({'variant': variant, 'features': len(feats.columns), **metrics})
        print(f'\n[{variant}] n_features={len(feats.columns)}'
              f'  accuracy={metrics["accuracy"]:.4f}'
              f'  balanced_accuracy={metrics["balanced_accuracy"]:.4f}'
              f'  incorrect_recall={metrics["incorrect_recall"]:.4f}'
              f'  incorrect_f1={metrics["incorrect_f1"]:.4f}')

    pd.DataFrame(summary).to_csv(args.out_dir / 'comparison.csv', index=False)
    prediction_table.to_csv(args.out_dir / 'lopo_predictions.csv', index=False)
    df.to_csv(args.out_dir / 'used_repetitions.csv', index=False)
    matrices['Extended'].to_csv(args.out_dir / 'extended_features.csv', index=False)
    # Track observability separately from predictive scores. For example,
    # knee-width ratios may be unavailable in side/half-profile views.
    coverage = pd.DataFrame({
        'feature': matrices['Extended'].columns,
        'observed_fraction': matrices['Extended'].notna().mean(axis=0).to_numpy(),
    })
    coverage['group'] = coverage.feature.str.split('_').str[0].replace({
        'asym': 'asymmetry', 'align': 'alignment', 'traj': 'trajectory',
    })
    coverage.to_csv(args.out_dir / 'feature_coverage.csv', index=False)
    print('\nMean observed feature fraction by group:')
    print(coverage.groupby('group')['observed_fraction'].mean().round(3).to_string())
    alignment_by_video = df[['video_id', 'person_id', 'repetition_number']].copy()
    alignment_by_video['width_usable_fraction'] = matrices['Extended']['align_width_usable_fraction']
    alignment_by_video.groupby('video_id')['width_usable_fraction'].agg(['mean', 'min', 'max']).to_csv(
        args.out_dir / 'alignment_coverage_by_video.csv')

    old = out_report['variants']['Baseline160']['pooled_out_of_person']
    new = out_report['variants']['Extended']['pooled_out_of_person']
    print('\nEXTENDED minus BASELINE (percentage points):')
    for k in ('accuracy', 'balanced_accuracy', 'incorrect_recall', 'incorrect_f1', 'macro_f1'):
        print(f'  {k}: {(new[k]-old[k])*100:+.2f} pp')

    # Print error changes per video, including both newly detected and newly
    # missed incorrect reps, to prevent average metrics masking weak subjects.
    delta = prediction_table.copy()
    delta['base_ok'] = (delta.label == delta.pred_Baseline160)
    delta['ext_ok'] = (delta.label == delta.pred_Extended)
    delta['outcome'] = np.select(
        [~delta.base_ok & delta.ext_ok, delta.base_ok & ~delta.ext_ok],
        ['fixed_by_extended', 'broken_by_extended'], default='unchanged')
    delta['base_missed_incorrect'] = ((delta.label == 0) & (delta.pred_Baseline160 == 1))
    delta['ext_missed_incorrect'] = ((delta.label == 0) & (delta.pred_Extended == 1))
    delta.to_csv(args.out_dir / 'error_changes.csv', index=False)
    print('\nChanged predictions:')
    print(delta[delta.outcome != 'unchanged'][[
        'video_id', 'repetition_number', 'label', 'pred_Baseline160',
        'pred_Extended', 'outcome']].to_string(index=False))
    print('\nFalse negatives by video (baseline -> extended):')
    for vid, g in delta.groupby('video_id', sort=True):
        a, b = int(g.base_missed_incorrect.sum()), int(g.ext_missed_incorrect.sum())
        print(f'  {vid}: {a} -> {b}')

    # Save an exploratory candidate; original baseline artifact stays untouched.
    # Only baseline/extended are eligible, regardless of --ablation.
    selected = ('Extended' if new['balanced_accuracy'] > old['balanced_accuracy'] else 'Baseline160')
    feats = matrices[selected]
    final = clone(template).fit(feats.to_numpy(dtype=float), y)
    joblib.dump({
        'pipeline': final,
        'feature_names': feats.columns.tolist(),
        'feature_groups': list(choices[selected]),
        'feature_variant': selected,
        'label_map': {0: 'incorrect', 1: 'correct'},
        'extractor_module': 'train_squat_extended.py',
        'source': 'REHAB24-6 YOLOv8m pose tracked camera17',
    }, args.out_dir / 'squat_classifier_candidate.joblib')
    out_report['candidate_model'] = selected
    with (args.out_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(out_report, f, ensure_ascii=False, indent=2)
    print(f'\nCandidate model: {selected} (exploratory selection, NOT independent test).')
    print('Saved to:', args.out_dir.resolve())
    print('NOTE: no clinical error type can be predicted from binary labels alone.')


if __name__ == '__main__':
    main()
