#!/usr/bin/env python3
"""Compare complete annotated repetitions vs stream-cropped repetitions.

Uses the SAME saved offline 2D keypoints for both variants to isolate the
impact of boundary cropping. Compares those outcomes with recorded online
results, which may additionally differ because of per-frame pose extraction.

Analysis only. Ground-truth boundaries are never fed to the online detector.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def iou_interval(a0, a1, b0, b1):
    n = max(0, min(a1, b1) - max(a0, b0) + 1)
    d = (a1 - a0 + 1) + (b1 - b0 + 1) - n
    return n / d if d > 0 else 0.0


def predict(seq, artifact, extractor):
    groups = extractor(seq)
    feats = dict(groups['base'])
    feats.update(groups['asymmetry'])
    names = artifact['feature_names']
    if len(names) != 180 or set(names) != set(feats):
        raise RuntimeError('Unexpected 180-feature schema for candidate model')
    x = np.asarray([[feats[n] for n in names]], dtype=np.float64)
    model = artifact['pipeline']
    cls = int(model.predict(x)[0])
    classes = list(model.classes_)
    prob_good = float(model.predict_proba(x)[0][classes.index(1)])
    return cls, prob_good


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--video', default='PM_043')
    pa.add_argument('--data-dir', type=Path,
                    default=Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'))
    pa.add_argument('--model', type=Path,
                    default=Path('models/squat_plus_asymmetry180.joblib'))
    pa.add_argument('--events', type=Path, default=None)
    pa.add_argument('--output', type=Path, default=None)
    pa.add_argument('--min-iou', type=float, default=.2)
    args = pa.parse_args()
    args.events = args.events or Path(f'results/stream_test/{args.video}_events.jsonl')
    args.output = args.output or Path(f'results/stream_test/{args.video}_boundary_diagnostic.csv')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_squat_extended import extract_feature_groups

    artifact = joblib.load(args.model)  # Only use artifact from the user's own training.
    if artifact.get('feature_variant') != 'PlusAsymmetry180':
        raise ValueError('The saved model must be the PlusAsymmetry180 artifact')
    idx = pd.read_csv(args.data_dir / 'index.csv')
    subset = idx[idx.video_id == args.video].sort_values('repetition_number')
    if subset.empty:
        raise RuntimeError(f'No extracted repetitions for {args.video}')
    events = []
    with args.events.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get('type') == 'rep':
                    events.append(item)
    events.sort(key=lambda v: (int(v['start_frame']), int(v['end_frame'])))

    offline = []
    for row in subset.itertuples(index=False):
        with np.load(args.data_dir / str(row.file), allow_pickle=False) as z:
            ks = z['keypoints'].copy()
            ids = z['frame_indices'].copy().astype(int)
        if len(ids) == 0 or len(ks) != len(ids):
            raise ValueError(f'Invalid stored sequence {row.file}')
        offline.append((row, ks, ids))

    candidates = []
    for i, (_, _, ids) in enumerate(offline):
        for j, ev in enumerate(events):
            score = iou_interval(int(ids[0]), int(ids[-1]),
                                 int(ev['start_frame']), int(ev['end_frame']))
            if score >= args.min_iou:
                candidates.append((score, i, j))
    candidates.sort(reverse=True)
    matched = {}
    used_events = set()
    for overlap, i, j in candidates:
        if i not in matched and j not in used_events:
            matched[i] = (j, overlap)
            used_events.add(j)

    rows = []
    for i, (meta, kp, ids) in enumerate(offline):
        if i not in matched:
            print(f'Unmatched annotated rep {meta.repetition_number}: {ids[0]}-{ids[-1]}')
            continue
        j, overlap = matched[i]
        event = events[j]
        mask = (ids >= int(event['start_frame'])) & (ids <= int(event['end_frame']))
        if mask.sum() < 10:
            print(f'Too few cropped frames rep {meta.repetition_number}: {int(mask.sum())}')
            continue
        full_pred, full_prob = predict(kp, artifact, extract_feature_groups)
        crop_pred, crop_prob = predict(kp[mask], artifact, extract_feature_groups)
        stream_posture = event.get('posture', 'unknown')
        online_pred = {'good':1, 'bad':0}.get(stream_posture, -1)
        gt = int(meta.label)
        rows.append({
            'video_id':args.video, 'rep':int(meta.repetition_number),
            'gt':gt,
            'offline_full_pred':full_pred,
            'offline_cropped_pred':crop_pred,
            'stream_pred':online_pred,
            'full_prob_correct':round(full_prob, 4),
            'cropped_prob_correct':round(crop_prob, 4),
            'stream_prob_correct':event.get('probability_correct_uncalibrated'),
            'gt_start':int(ids[0]), 'gt_end':int(ids[-1]),
            'det_start':int(event['start_frame']),
            'det_end':int(event['end_frame']),
            'overlap_iou':round(overlap, 3),
            'retained_frame_fraction':round(float(mask.mean()), 3),
            'prediction_changed_by_crop':int(full_pred != crop_pred),
            'offline_cropped_vs_stream_disagree':int(crop_pred != online_pred),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError('No matched diagnostic samples')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Video {args.video}: annotated={len(offline)}, stream_events={len(events)}, matches={len(result)}')
    print(f'Mean retained fraction of original GT frames: {result.retained_frame_fraction.mean():.3f}')
    for col in ('offline_full_pred','offline_cropped_pred','stream_pred'):
        keep = result[col] >= 0
        acc = (result.loc[keep,col] == result.loc[keep,'gt']).mean()
        incorrect = keep & (result["gt"] == 0)
        recall = (result.loc[incorrect,col] == 0).mean() if incorrect.any() else float('nan')
        print(f'{col:25s}  matched_accuracy={acc:.3f}  incorrect_recall={recall:.3f}')
    print(f'GT-full vs GT-cropped disagreements: {int(result.prediction_changed_by_crop.sum())}')
    print(f'GT-cropped vs stream disagreements: {int(result.offline_cropped_vs_stream_disagree.sum())}')
    print('\nIncorrect reps (0=bad, 1=good):')
    print(result[result["gt"]==0][['rep','gt','offline_full_pred','offline_cropped_pred','stream_pred',
                                'full_prob_correct','cropped_prob_correct','retained_frame_fraction']]
          .to_string(index=False))
    print(f'\nSaved: {args.output}')
    print('Caution: this is an in-sample diagnosis, NOT an unseen-person performance estimate.')
    print('Offline-cropped vs stream also reflects differences in YOLO/tracking extraction.')


if __name__ == '__main__':
    main()
