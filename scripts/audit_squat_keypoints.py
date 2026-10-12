"""Visual and numeric audit of tracked REHAB24-6 Camera17 squat keypoints.

Run from ~/capstone2/pose_training:
    python scripts/audit_squat_keypoints.py

Requires: opencv-python, numpy (existing capstone_pose environment).
Does NOT run YOLO or change the original .npz files.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('data/raw/REHAB24-6')
DATA = Path('data/processed/REHAB24-6/camera17_yolov8m_tracked')
DEST = Path('results/squat_quality_audit')
CONF_T = 0.30
THUMB_W, THUMB_H = 400, 225
# COCO-17 pose skeleton (0-based)
EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


def load_data(path):
    with np.load(path, allow_pickle=False) as obj:
        return {k: obj[k].copy() for k in obj.files}


def max_jump(centers, valid):
    if len(centers) < 2:
        return float('nan')
    pair_ok = valid[1:] & valid[:-1]
    dist = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return float(np.max(dist[pair_ok])) if pair_ok.any() else float('nan')


def stats(d):
    kp = d['keypoints']
    boxes = d['selected_boxes']
    valid_kp = np.isfinite(kp[:, :, :2]).all(axis=2) & (kp[:, :, 2] >= CONF_T)
    found = np.isfinite(boxes).all(axis=1)
    hips = kp[:, [11, 12], :2]
    hip_valid = valid_kp[:, 11] & valid_kp[:, 12]
    hip_centers = hips.mean(axis=1)
    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    return {
        'valid_keypoint_ratio': float(valid_kp.mean()),
        'target_missing_frames': int((~found).sum()),
        'multi_person_frames': int((d['person_counts'] > 1).sum()),
        'max_hip_jump_px': max_jump(hip_centers, hip_valid),
        'max_box_center_jump_px': max_jump(box_centers, found),
    }


def frame_at(cap, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'Failed to read frame {n}')
    return frame


def draw_pose(frame, kp, box):
    im = frame.copy()
    if np.isfinite(box).all():
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 4)
    visible = np.isfinite(kp[:, :2]).all(axis=1) & (kp[:, 2] >= CONF_T)
    for a, b in EDGES:
        if visible[a] and visible[b]:
            p = tuple(np.rint(kp[a, :2]).astype(int))
            q = tuple(np.rint(kp[b, :2]).astype(int))
            cv2.line(im, p, q, (0, 255, 255), 4, cv2.LINE_AA)
    for i in np.flatnonzero(visible):
        p = tuple(np.rint(kp[i, :2]).astype(int))
        cv2.circle(im, p, 5, (255, 0, 255), -1, cv2.LINE_AA)
    return im


def tile_from_frame(cap, d, row, position, phase):
    indices = d['frame_indices']
    n = int(position)
    if not 0 <= n < len(indices):
        raise IndexError(f'Invalid position {n}')
    frame = frame_at(cap, indices[n])
    frame = draw_pose(frame, d['keypoints'][n], d['selected_boxes'][n])
    thumbnail = cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    text = f"{row['video_id']} rep{int(row['repetition_number']):02d} {phase} "
    text += f"f{int(indices[n])}  {'OK' if row['correctness']=='1' else 'BAD'}"
    cv2.rectangle(thumbnail, (0, 0), (THUMB_W, 29), (24, 24, 24), -1)
    cv2.putText(thumbnail, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if not np.isfinite(d['selected_boxes'][n]).all():
        cv2.putText(thumbnail, 'TARGET MISSING', (8, THUMB_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return thumbnail


def save_sheet(tiles, columns, file):
    if not tiles:
        return
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * THUMB_H, columns * THUMB_W, 3), np.uint8)
    for i, thumb in enumerate(tiles):
        y, x = divmod(i, columns)
        sheet[y * THUMB_H:(y + 1) * THUMB_H,
              x * THUMB_W:(x + 1) * THUMB_W] = thumb
    if not cv2.imwrite(str(file), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f'Could not write {file}')
    print('Saved:', file)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'Segmentation.csv').open(newline='') as f:
        squat_rows = [r for r in csv.DictReader(f, delimiter=';') if r['exercise_id'] == '6']
    if len(squat_rows) != 195:
        raise ValueError(f'Expected 195 Squat repetitions, got {len(squat_rows)}')

    groups = defaultdict(list)
    reports = []
    for row in squat_rows:
        video_id = row['video_id']
        rep = int(row['repetition_number'])
        file = DATA / f'{video_id}_rep{rep:02d}.npz'
        if not file.is_file():
            raise FileNotFoundError(file)
        d = load_data(file)
        expected_indices = np.arange(int(row['first_frame']), int(row['last_frame']) + 1)
        if d['keypoints'].shape != (len(expected_indices), 17, 3):
            raise ValueError(f'Wrong keypoint shape: {file}')
        if d['selected_boxes'].shape != (len(expected_indices), 4):
            raise ValueError(f'Wrong bbox shape: {file}')
        if not np.array_equal(d['frame_indices'], expected_indices):
            raise ValueError(f'Frame mismatch: {file}')
        if int(d['label']) != int(row['correctness']) or int(d['person_id']) != int(row['person_id']):
            raise ValueError(f'Label / person mismatch: {file}')
        s = stats(d)
        flag = (s['valid_keypoint_ratio'] < 0.90 or s['target_missing_frames'] > 0
                or s['max_hip_jump_px'] > 60 or s['max_box_center_jump_px'] > 120)
        reports.append({
            'video_id': video_id, 'rep': rep, 'person_id': int(row['person_id']),
            'label': int(row['correctness']), 'frames': len(expected_indices),
            'valid_ratio': round(s['valid_keypoint_ratio'], 5),
            'target_missing': s['target_missing_frames'],
            'multi_person': s['multi_person_frames'],
            'max_hip_jump_px': round(s['max_hip_jump_px'], 2),
            'max_box_center_jump_px': round(s['max_box_center_jump_px'], 2),
            'review_flag': int(flag),
        })
        groups[video_id].append(row)

    output_csv = DEST / 'audit.csv'
    with output_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(reports[0].keys()))
        writer.writeheader()
        writer.writerows(reports)
    print('Audit CSV:', output_csv)
    print('Repetitions:', len(reports))
    print('Flagged reps:', sum(a['review_flag'] for a in reports))
    print('Missing target frames:', sum(a['target_missing'] for a in reports))
    print('Mean valid ratio:', round(np.mean([a['valid_ratio'] for a in reports]), 4))
    print('Max hip jump:', max(a['max_hip_jump_px'] for a in reports), 'px')
    print('Max box-center jump:', max(a['max_box_center_jump_px'] for a in reports), 'px')
    flagged = [a for a in reports if a['review_flag']]
    for a in flagged[:30]:
        print('REVIEW:', a['video_id'], 'rep', a['rep'],
              'valid=', a['valid_ratio'], 'missing=', a['target_missing'],
              'hip_jump=', a['max_hip_jump_px'], 'box_jump=', a['max_box_center_jump_px'])
    if len(flagged) > 30:
        print('... additional flagged repetitions:', len(flagged) - 30)

    # All 195 repetition midpoints, plus early/middle/late repetition phase sheets.
    for video_id, rows in sorted(groups.items()):
        rows.sort(key=lambda x: int(x['repetition_number']))
        video_path = ROOT / f'{video_id}-Camera17-30fps.mp4'
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {video_path}')
        try:
            midpoint_tiles = []
            for row in rows:
                d = load_data(DATA / f"{video_id}_rep{int(row['repetition_number']):02d}.npz")
                midpoint_tiles.append(tile_from_frame(cap, d, row,
                                                       len(d['frame_indices']) // 2, 'MID'))
            save_sheet(midpoint_tiles, 5, DEST / f'all_reps_{video_id}.jpg')
            # 3 repetitions throughout the recording, each with start/middle/end frames.
            n = len(rows)
            three_rows = [rows[i] for i in sorted(set([0, n // 2, n - 1]))]
            phase_tiles = []
            for row in three_rows:
                d = load_data(DATA / f"{video_id}_rep{int(row['repetition_number']):02d}.npz")
                T = len(d['frame_indices'])
                for phase, pos in [('START', 0), ('MID', T // 2), ('END', T - 1)]:
                    phase_tiles.append(tile_from_frame(cap, d, row, pos, phase))
            save_sheet(phase_tiles, 3, DEST / f'phases_{video_id}.jpg')
        finally:
            cap.release()

    print('\nDone. Inspect ALL nine all_reps_*.jpg sheets and nine phases_*.jpg sheets.')
    print('Red box = chosen person. Yellow = skeleton, magenta = joints.')
    print('Flagged repetitions are review candidates, not automatic failures.')


if __name__ == '__main__':
    main()
