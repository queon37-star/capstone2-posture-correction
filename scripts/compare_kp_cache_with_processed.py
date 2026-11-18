"""Consistency check: full-video cache vs the training repetition NPZ files.

The training NPZs were extracted per repetition (target tracking restarted at
each rep start, batch-16 inference, conf<0.3 masked). The cache tracks
continuously from frame 0 with per-frame inference. This reports how often the
two pick a different person and how far the keypoints differ on GT rep frames,
so cache-based results can be related to the original training inputs.
Read-only.

  python scripts/compare_kp_cache_with_processed.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_detect_hailo_bridge import read_kp_dump  # noqa: E402


def compare(cache_kp, npz_kp, mask_conf=0.30):
    c = cache_kp.copy()
    c[c[:, :, 2] < mask_conf, :2] = np.nan
    c_person = np.isfinite(c[:, :, :2]).any(axis=(1, 2))
    n_person = np.isfinite(npz_kp[:, :, :2]).any(axis=(1, 2))
    both = c_person & n_person
    finite = np.isfinite(c[:, :, :2]) & np.isfinite(npz_kp[:, :, :2])
    diff = np.abs(c[:, :, :2] - npz_kp[:, :, :2])[finite]
    far = np.zeros(len(c), bool)               # selected a different person (> 50 px median)
    for i in np.flatnonzero(both):
        f = finite[i].all(axis=1)
        if f.any():
            far[i] = np.median(np.abs(c[i, f, :2] - npz_kp[i, f, :2])) > 50
    return {'frames': int(len(c)), 'presence_mismatch': int((c_person != n_person).sum()),
            'different_person_frames': int(far.sum()),
            'joint_nan_pattern_mismatch': float(np.mean(np.isfinite(c[:, :, 0]) != np.isfinite(npz_kp[:, :, 0]))),
            'xy_abs_diff_mean': float(diff.mean()) if diff.size else None,
            'xy_abs_diff_p99': float(np.percentile(diff, 99)) if diff.size else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', type=Path, default=Path('data/cache/kp_full/rehab24_camera17_yolov8m'))
    ap.add_argument('--data-dir', type=Path, default=Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'))
    ap.add_argument('--out', type=Path, default=None, help='optional JSON report path')
    args = ap.parse_args()
    index = pd.read_csv(args.data_dir / 'index.csv')
    report = {}
    for video, rows in index.groupby('video_id'):
        cache_path = args.cache_dir / f'{video}.kp.bin'
        if not cache_path.is_file():
            report[video] = 'no cache'
            continue
        data = read_kp_dump(cache_path)
        totals = []
        for r in rows.itertuples(index=False):
            with np.load(args.data_dir / r.file, allow_pickle=False) as pack:
                npz_kp = pack['keypoints']
            s, e = int(r.start_frame), int(r.end_frame)
            if e >= len(data):
                totals.append({'rep': int(r.repetition_number), 'error': 'beyond cache'})
                continue
            totals.append({'rep': int(r.repetition_number), **compare(data['kp'][s:e + 1], npz_kp)})
        ok = [t for t in totals if 'error' not in t]
        report[video] = {
            'reps': len(totals), 'frames': sum(t['frames'] for t in ok),
            'presence_mismatch': sum(t['presence_mismatch'] for t in ok),
            'different_person_frames': sum(t['different_person_frames'] for t in ok),
            'reps_with_person_difference': [t['rep'] for t in ok if t['different_person_frames']],
            'xy_abs_diff_mean': float(np.nanmean([t['xy_abs_diff_mean'] for t in ok
                                                  if t['xy_abs_diff_mean'] is not None] or [np.nan])),
            'errors': [t for t in totals if 'error' in t]}
        print(video, json.dumps(report[video]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
