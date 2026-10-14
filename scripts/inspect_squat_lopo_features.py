#!/usr/bin/env python3
"""Diagnose LOPO logistic-regression feature evidence for full/cropped squat reps.

This is a post-hoc model diagnostic, NOT an accuracy measurement on new users.
It reads held-out GT keypoint NPZs and already-generated diagnosis CSVs. It does
NOT change the classifier or the online repetition detector.

For binary LR with classes [0,1], positive contributions favor 'good'.
Feature contribution is algebraic (standardized feature * LR coefficient),
not proof of causal influence or reliable physiological interpretation.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

def vector_for(seq, names):
    from train_squat_extended import extract_feature_groups
    g = extract_feature_groups(seq)
    d = {**g['base'], **g['asymmetry']}
    if len(names) != 180 or set(d) != set(names):
        raise ValueError('PlusAsymmetry180 feature schema mismatch')
    return np.array([d[n] for n in names], dtype=np.float64)


def explain_vector(pipeline, values):
    """Return signed evidence for Good (class 1), in log-odds units."""
    classes = list(pipeline.classes_)
    if classes != [0, 1]:
        raise ValueError(f'Expected classes [0,1], got {classes}')
    imputer = pipeline.named_steps['imputer']
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    x = np.where(np.isfinite(values), values, np.nan).reshape(1, -1)
    z = scaler.transform(imputer.transform(x))[0]
    coef = np.asarray(model.coef_).reshape(-1)
    if len(z) != len(coef):
        raise RuntimeError('Pipeline feature dimension mismatch')
    evidence = z * coef
    intercept = float(model.intercept_[0])
    decision = intercept + float(evidence.sum())
    reference = float(np.asarray(pipeline.decision_function(x)).reshape(-1)[0])
    if not np.isclose(decision, reference, atol=1e-7, rtol=1e-7):
        raise RuntimeError('Feature evidence does not reconstruct decision_function')
    p_good = float(pipeline.predict_proba(x)[0, classes.index(1)])
    pred = int(pipeline.predict(x)[0])
    return evidence, intercept, decision, p_good, pred


def top(evidence, names, sense, k=4):
    """Most influential signed terms in a selected direction (not causal importance)."""
    arr = sense * evidence
    order = np.argsort(-arr)
    return ', '.join(f'{names[j]}({evidence[j]:+.2f})' for j in order[:k])


def analyse(video, data_dir, model_dir, diagnosis_dir, out_dir, print_limit):
    art = joblib.load(model_dir / f'{video}.joblib')  # Trust only own locally produced artifact.
    if art.get('feature_variant') != 'PlusAsymmetry180' or art.get('held_out_video') != video:
        raise ValueError(f'Not the expected held-out 180-feature model: {video}')
    names = art['feature_names']
    pipe = art['pipeline']
    diag = pd.read_csv(diagnosis_dir / f'{video}_boundary_diagnostic.csv')
    idx = pd.read_csv(data_dir / 'index.csv')
    samples = idx.loc[idx['video_id'] == video]
    mapping = {int(r['repetition_number']): r for _, r in samples.iterrows()}
    if len(mapping) != len(diag):
        raise ValueError(f'Index and diagnostic repetition counts differ for {video}')
    data = []
    for d in diag.itertuples(index=False):
        meta = mapping[int(d.rep)]
        with np.load(data_dir / str(meta['file']), allow_pickle=False) as sample:
            kp = sample['keypoints'].copy()
            ids = sample['frame_indices'].astype(int)
        mask = (ids >= int(d.det_start)) & (ids <= int(d.det_end))
        if mask.sum() < 10:
            raise ValueError(f'{video} rep{d.rep}: too few cropped frames')
        full_v = vector_for(kp, names)
        crop_v = vector_for(kp[mask], names)
        full_e, intercept, full_score, full_prob, full_pred = explain_vector(pipe, full_v)
        crop_e, _, crop_score, crop_prob, crop_pred = explain_vector(pipe, crop_v)
        if full_pred != int(d.offline_full_pred) or crop_pred != int(d.offline_cropped_pred):
            raise AssertionError(f'{video} rep{d.rep}: diagnostic prediction mismatch; check model/NPZ')
        gt = int(d.gt)
        online = int(d.stream_pred)
        info = {
            'video': video, 'rep': int(d.rep), 'gt': gt,
            'full_pred': full_pred, 'crop_pred': crop_pred, 'stream_pred': online,
            'full_prob_good': round(full_prob, 5), 'crop_prob_good': round(crop_prob, 5),
            'full_log_odds_good': round(full_score, 4), 'crop_log_odds_good': round(crop_score, 4),
            'crop_log_odds_shift': round(crop_score - full_score, 4),
            'retained_gt_fraction': float(d.retained_frame_fraction),
            'full_wrong': int(full_pred != gt),
            'crop_flip': int(crop_pred != full_pred),
            'stream_flip': int(online != crop_pred),
            'evidence_favoring_wrong_full': (top(full_e, names, +1 if gt == 0 else -1)
                                              if full_pred != gt else ''),
            'strongest_crop_changes': ', '.join(
                f'{names[j]}({(crop_e-full_e)[j]:+.2f})'
                for j in np.argsort(-np.abs(crop_e-full_e))[:4]
            ),
        }
        data.append(info)
    result = pd.DataFrame(data).sort_values('rep')
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / f'{video}_feature_errors.csv', index=False)
    print(f'\n===== {video} =====')
    print('Full/Cropped/Stream:',
          f"{int((result.full_pred==result['gt']).sum())}/{len(result)}",
          f"{int((result.crop_pred==result['gt']).sum())}/{len(result)}",
          f"{int((result.stream_pred==result['gt']).sum())}/{len(result)}")
    print(f"GT-full wrong={int(result.full_wrong.sum())}, crop flips={int(result.crop_flip.sum())}, "
          f"crop-to-stream flips={int(result.stream_flip.sum())}")
    print('Mistakes (full wrong or stream wrong), limited display:')
    shown = result[(result.full_wrong == 1) | (result.stream_pred != result["gt"])].head(print_limit)
    for d in shown.itertuples(index=False):
        print(f'  #{d.rep:02d} GT={d.gt} full={d.full_pred} cropped={d.crop_pred} '
              f'stream={d.stream_pred} p_good(full,crop)=({d.full_prob_good:.3f},{d.crop_prob_good:.3f})')
        if d.full_wrong:
            print('      Main evidence toward WRONG class (full):', d.evidence_favoring_wrong_full)
        if d.crop_flip:
            print('      Largest full→crop changes:', d.strongest_crop_changes)
    print('Saved:', out_dir / f'{video}_feature_errors.csv')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', nargs='+', default=['PM_022', 'PM_043', 'PM_113'])
    ap.add_argument('--data-dir', type=Path,
                    default=Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'))
    ap.add_argument('--model-dir', type=Path, default=Path('models/squat_lopo_stream'))
    ap.add_argument('--diagnosis-dir', type=Path, default=Path('results/stream_lopo_eval'))
    ap.add_argument('--out-dir', type=Path, default=Path('results/stream_lopo_feature_diagnostics'))
    ap.add_argument('--print-limit', type=int, default=12)
    args = ap.parse_args()
    for video in args.videos:
        analyse(video, args.data_dir, args.model_dir, args.diagnosis_dir,
                args.out_dir, args.print_limit)
    print('\nEvidence is algebraic LR attribution, not causality. No retraining or test-data tuning done.')


if __name__ == '__main__':
    main()
