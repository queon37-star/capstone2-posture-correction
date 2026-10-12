"""REHAB24-6 squat YOLOv8m-Pose extraction with fixed ROI + temporal identity.

Original dataset outputs remain untouched. Supports --video and --rep for a
small validation run, and resumes existing (T,17,3) extraction files.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE = Path('data/raw/REHAB24-6')
CONFIG = Path('configs/rehab24_squat_camera17_subjects.json')
OUT = Path('data/processed/REHAB24-6/camera17_yolov8m_tracked')
DEBUG = Path('results/person_selection_tracked')
BATCH_SIZE = 16
KP_CONF_THRESHOLD = 0.30


def box_center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def box_iou(a, b):
    ix1, iy1 = np.maximum(a[:2], b[:2])
    ix2, iy2 = np.minimum(a[2:], b[2:])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    aa = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    ab = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-6)


def select_target(boxes, scores, anchor, prev):
    """Return selected detection index, or None if no candidate matches ROI.

    The manually designated anchor defines a safe fixed spatial region.
    A previous selected bbox adds temporal consistency. Confidence is a tie
    breaker, never the sole selection criterion.
    """
    if len(boxes) == 0:
        return None
    ac = box_center(anchor)
    size = np.maximum(anchor[2:] - anchor[:2], [1., 1.])
    best_idx, best_cost = None, float('inf')
    for i, (box, conf) in enumerate(zip(boxes, scores)):
        c = box_center(box)
        relative = np.abs(c - ac) / size
        # Squats move vertically, but the main exercise position is stationary.
        if relative[0] > 0.95 or relative[1] > 0.75:
            continue
        d_ref = np.linalg.norm((c - ac) / size)
        if prev is None:
            cost = 2.0 * d_ref - 0.10 * conf
        else:
            d_prev = np.linalg.norm((c - box_center(prev)) / size)
            overlap = box_iou(box, prev)
            # Reject a discontinuous identity switch in a single frame.
            if d_prev > 0.75 and overlap < 0.08:
                continue
            cost = 1.6 * d_ref + 1.0 * d_prev - 1.2 * overlap - 0.10 * conf
        if cost < best_cost:
            best_idx, best_cost = i, cost
    return best_idx


def analyze(sequence):
    kp_conf = sequence[:, :, 2]
    valid_ratio = float((kp_conf >= KP_CONF_THRESHOLD).mean())
    hips = sequence[:, [11, 12], :2]
    ok = np.isfinite(hips).all(axis=(1, 2)) & (kp_conf[:, [11, 12]] >= KP_CONF_THRESHOLD).all(axis=1)
    centers = hips.mean(axis=1)
    dist = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    dist = dist[ok[:-1] & ok[1:]]
    return valid_ratio, float(dist.max()) if len(dist) else float('nan')


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', help='e.g. PM_022 (default: all videos)')
    ap.add_argument('--rep', type=int, help='one repetition (requires --video)')
    ap.add_argument('--overwrite', action='store_true', help='force re-extraction')
    args = ap.parse_args()
    if args.rep is not None and not args.video:
        ap.error('--rep requires --video')
    if not CONFIG.exists():
        raise FileNotFoundError(f'First choose subject anchors using select_subjects.py: {CONFIG}')
    anchors = json.loads(CONFIG.read_text())
    with (BASE / 'Segmentation.csv').open(newline='') as f:
        all_rows = list(csv.DictReader(f, delimiter=';'))
    squat_rows = [r for r in all_rows if r['exercise_id'] == '6']
    if args.video:
        squat_rows = [r for r in squat_rows if r['video_id'] == args.video]
    if args.rep is not None:
        squat_rows = [r for r in squat_rows if int(r['repetition_number']) == args.rep]
    if not squat_rows:
        raise ValueError('No matching squat repetitions')
    videos = sorted(set(r['video_id'] for r in squat_rows))
    missing = [v for v in videos if v not in anchors]
    if missing:
        raise RuntimeError(f'Missing reference subject for {missing}. Run select_subjects.py first.')
    OUT.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)
    by_video = defaultdict(list)
    for row in squat_rows:
        by_video[row['video_id']].append(row)
    model = YOLO('yolov8m-pose.pt')
    metadata = []

    for video_id in videos:
        video_path = BASE / f'{video_id}-Camera17-30fps.mp4'
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {video_path}')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        anchor = np.asarray(anchors[video_id]['bbox'], dtype=np.float32)
        cursor = 0
        print(f'\n=== {video_id} ===', flush=True)
        for row in sorted(by_video[video_id], key=lambda r: int(r['first_frame'])):
            rep = int(row['repetition_number'])
            start, end = int(row['first_frame']), int(row['last_frame'])
            expected = end - start + 1
            assert 0 <= start <= end < total_frames, f'{video_id} rep {rep}: bad frame range'
            assert cursor <= start, f'{video_id} rep {rep}: overlapping frame ranges'
            while cursor < start:
                if not cap.grab():
                    raise RuntimeError(f'{video_id}: failed skipping frame {cursor}')
                cursor += 1
            out_path = OUT / f'{video_id}_rep{rep:02d}.npz'
            if out_path.exists() and not args.overwrite:
                with np.load(out_path) as saved:
                    sequence = saved['keypoints'].copy()
                    raw_count = saved['person_counts'].copy()
                    selected = saved['selected_boxes'].copy()
                assert sequence.shape == (expected, 17, 3), f'Mismatch: {out_path}'
                while cursor <= end:
                    if not cap.grab():
                        raise RuntimeError(f'{video_id}: failed skipping frame {cursor}')
                    cursor += 1
                print(f'  Rep {rep:02d}: reused')
            else:
                prev = None
                seq, counts, chosen_boxes = [], [], []
                missing_frames = 0
                while cursor <= end:
                    frames, indices = [], []
                    for _ in range(min(BATCH_SIZE, end - cursor + 1)):
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(f'{video_id}: failed reading frame {cursor}')
                        frames.append(frame)
                        indices.append(cursor)
                        cursor += 1
                    results = model.predict(frames, device=0, imgsz=640, conf=0.25, verbose=False)
                    for frame, index, result in zip(frames, indices, results):
                        box_count = len(result.boxes) if result.boxes is not None else 0
                        counts.append(box_count)
                        boxes = result.boxes.xyxy.cpu().numpy() if box_count else np.empty((0, 4))
                        scores = result.boxes.conf.cpu().numpy() if box_count else np.empty((0,))
                        chosen = select_target(boxes, scores, anchor, prev)
                        kp = np.zeros((17, 3), dtype=np.float32)
                        kp[:, :2] = np.nan
                        box = np.full(4, np.nan, dtype=np.float32)
                        if chosen is not None and result.keypoints is not None:
                            kp = result.keypoints.data[chosen].cpu().numpy().astype(np.float32)
                            kp[kp[:, 2] < KP_CONF_THRESHOLD, :2] = np.nan
                            box = boxes[chosen].astype(np.float32)
                            prev = box.copy()
                        else:
                            missing_frames += 1
                        seq.append(kp)
                        chosen_boxes.append(box)
                        if video_id == 'PM_022' and rep == 16 and index in (2139, 2140):
                            view = frame.copy()
                            for i, det_box in enumerate(boxes):
                                x1, y1, x2, y2 = det_box.astype(int)
                                chosen_now = i == chosen
                                color = (0, 0, 255) if chosen_now else (0, 255, 0)
                                cv2.rectangle(view, (x1, y1), (x2, y2), color, 4)
                                cv2.putText(view, 'SELECTED' if chosen_now else 'OTHER',
                                            (x1, max(40, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.0, color, 3)
                            dst = DEBUG / f'{video_id}_rep{rep:02d}_frame{index}.jpg'
                            cv2.imwrite(str(dst), view)
                sequence = np.stack(seq)
                raw_count = np.array(counts, dtype=np.int16)
                selected = np.stack(chosen_boxes)
                np.savez_compressed(
                    out_path, keypoints=sequence, frame_indices=np.arange(start, end + 1),
                    label=int(row['correctness']), person_id=int(row['person_id']),
                    video_id=video_id, repetition_number=rep, camera_id=17,
                    person_counts=raw_count, selected_boxes=selected,
                )
                print(f'  Rep {rep:02d}: {len(sequence)} frames; multi-person={(raw_count>1).sum()} '
                      f'frames; target missing={missing_frames}', flush=True)
            valid_ratio, max_hip_jump = analyze(sequence)
            metadata.append({
                'video_id': video_id, 'person_id': int(row['person_id']), 'repetition_number': rep,
                'start_frame': start, 'end_frame': end, 'orientation': row['cam17_orientation'],
                'label': int(row['correctness']), 'frames': len(sequence),
                'multi_person_frames': int((raw_count > 1).sum()),
                'target_found_frames': int(np.isfinite(selected).all(axis=1).sum()),
                'valid_keypoint_ratio': round(valid_ratio, 5),
                'max_hip_jump_px': round(max_hip_jump, 2), 'file': out_path.name,
            })
        cap.release()

    index_file = OUT / ('index.csv' if not args.video else f'index_{args.video}' +
                        (f'_rep{args.rep:02d}' if args.rep is not None else '') + '.csv')
    with index_file.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metadata[0]))
        w.writeheader()
        w.writerows(metadata)
    print('\n=== Completed ===')
    print('Repetitions:', len(metadata))
    print('Mean valid keypoint ratio:', round(np.mean([m['valid_keypoint_ratio'] for m in metadata]), 4))
    print('Max hip jump:', max(m['max_hip_jump_px'] for m in metadata), 'px')
    print('Index:', index_file)
    print('Debug JPGs:', DEBUG)


if __name__ == '__main__':
    main()
