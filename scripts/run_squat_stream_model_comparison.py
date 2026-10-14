#!/usr/bin/env python3
"""Run YOLO/target tracking/SquatCounter/RepBoundaryBuffer ONCE, then compare
three person-held-out classifiers on EXACTLY the same completed rep keypoints.

No annotation/Segmentation.csv or GT boundaries are read during inference.
This is an exploratory REHAB24-6 comparison, not independent external validation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import joblib
import numpy as np

VARIANTS = (
    'PlusAsymmetry180',
    'PlusAsymmetry180_crop05',
    'PlusAsymmetry180_crop10',
)


def load_artifacts(video, baseline_root, augmented_root):
    paths = {
        'PlusAsymmetry180': baseline_root / f'{video}.joblib',
        'PlusAsymmetry180_crop05': augmented_root / 'PlusAsymmetry180_crop05' / f'{video}.joblib',
        'PlusAsymmetry180_crop10': augmented_root / 'PlusAsymmetry180_crop10' / f'{video}.joblib',
    }
    artifacts = {}
    for variant, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{variant}: model missing: {path} (train with --save-models first)')
        artifact = joblib.load(path)  # Trust only artifacts trained locally by this project.
        if artifact.get('feature_variant') != variant:
            raise ValueError(f'{path}: expected feature_variant={variant!r}')
        if artifact.get('held_out_video') != video:
            raise ValueError(f'{path}: expected held_out_video={video!r}; got {artifact.get("held_out_video")!r}')
        names = artifact.get('feature_names')
        if not isinstance(names, list) or len(names) != 180 or len(set(names)) != 180:
            raise ValueError(f'{path}: expected 180 unique feature names')
        artifacts[variant] = artifact
    reference = set(artifacts[VARIANTS[0]]['feature_names'])
    if any(set(artifacts[v]['feature_names']) != reference for v in VARIANTS[1:]):
        raise ValueError('Models disagree on the input feature schema')
    return artifacts


def classify_three(rep_keypoints, artifacts, extract_feature_groups):
    groups = extract_feature_groups(rep_keypoints)
    record = {**groups['base'], **groups['asymmetry']}
    predictions = {}
    for variant, artifact in artifacts.items():
        names = artifact['feature_names']
        if set(record) != set(names):
            raise ValueError(f'{variant}: feature schema mismatch')
        x = np.array([[record[n] for n in names]], dtype=np.float64)
        model = artifact['pipeline']
        classes = list(model.classes_ if hasattr(model, 'classes_')
                       else model.named_steps['model'].classes_)
        if set(classes) != {0, 1}:
            raise ValueError(f'{variant}: expected labels 0=bad/1=good, got {classes}')
        pred = int(model.predict(x)[0])
        prob_good = float(model.predict_proba(x)[0, classes.index(1)])
        if not np.isfinite(prob_good):
            raise ValueError(f'{variant}: nonfinite probability')
        predictions[variant] = (pred, prob_good)
    return predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True)
    ap.add_argument('--video-file', type=Path)
    ap.add_argument('--anchor-config', type=Path,
                    default=Path('configs/rehab24_squat_camera17_subjects.json'))
    ap.add_argument('--baseline-model-root', type=Path, default=Path('models/squat_lopo_stream'))
    ap.add_argument('--augmented-model-root', type=Path,
                    default=Path('results/lopo_robustness_experiment/models'))
    ap.add_argument('--output-root', type=Path,
                    default=Path('results/stream_variant_comparison'))
    ap.add_argument('--yolo', default='yolov8m-pose.pt')
    ap.add_argument('--device', default='0')
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=0, help='0=entire video')
    ap.add_argument('--bottom-depth', type=float, default=0.20)
    ap.add_argument('--bottom-knee', type=float, default=120.0)
    ap.add_argument('--pre-frames', type=int, default=16)
    ap.add_argument('--post-frames', type=int, default=22)
    ap.add_argument('--paced', action='store_true')
    args = ap.parse_args()
    if args.start_frame < 0 or args.max_frames < 0:
        ap.error('--start-frame and --max-frames must be nonnegative')
    if not (0.16 < args.bottom_depth < 2.0):
        ap.error('--bottom-depth must be in (0.16, 2.0)')
    if not (0 < args.bottom_knee <= 180):
        ap.error('--bottom-knee must be in (0, 180]')
    if args.pre_frames < 0 or args.post_frames < 0:
        ap.error('context frame counts cannot be negative')

    artifacts = load_artifacts(args.video, args.baseline_model_root, args.augmented_model_root)
    # Imports delayed to keep the pure inference/multimodel helper unit-testable.
    import cv2
    from ultralytics import YOLO
    from extract_all_squats_tracked import select_target
    from run_squat_video_stream import absent
    from stream_squat_core import SquatCounter
    from squat_rep_buffer import RepBoundaryBuffer
    from train_squat_extended import extract_feature_groups

    video_path = args.video_file or Path(f'data/raw/REHAB24-6/{args.video}-Camera17-30fps.mp4')
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    config = json.loads(args.anchor_config.read_text(encoding='utf-8'))
    if args.video not in config:
        raise KeyError(f'No target person anchor found for {args.video}')
    anchor = np.array(config[args.video]['bbox'], dtype=np.float32)
    pose_model = YOLO(args.yolo)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    counter = SquatCounter(fps=fps, bottom_depth=args.bottom_depth,
                           bottom_knee=args.bottom_knee)
    buffer = RepBoundaryBuffer(fps=fps, pre_frames=args.pre_frames, post_frames=args.post_frames)
    previous_box = None
    n = chosen = published = 0
    t0 = time.perf_counter()

    with ExitStack() as stack:
        writers = {}
        for variant in VARIANTS:
            output = args.output_root / variant / f'{args.video}_events.jsonl'
            output.parent.mkdir(parents=True, exist_ok=True)
            writers[variant] = stack.enter_context(output.open('w', encoding='utf-8'))

        def emit(event, publish_frame):
            nonlocal published
            preds = classify_three(event.keypoints, artifacts, extract_feature_groups)
            published += 1
            for variant in VARIANTS:
                pred, prob_good = preds[variant]
                msg = {
                    'type': 'rep', 'exercise': 'squat', 'count': published,
                    'posture': 'good' if pred == 1 else 'bad',
                    'issues': [], 'score': None,
                    'probability_correct_uncalibrated': round(prob_good, 4),
                    'start_frame': event.start_frame, 'end_frame': event.end_frame,
                    'detected_start_frame': event.detected_start_frame,
                    'detected_end_frame': event.detected_end_frame,
                    'classification_frame_count': len(event.keypoints),
                    'post_context_complete': event.post_context_complete,
                    'peak_hip_depth_torso_units': round(event.peak_depth, 3),
                    'min_projected_knee_angle': round(event.min_knee_angle, 1),
                    'model_variant': variant,
                    'publish_frame': publish_frame,
                }
                writers[variant].write(json.dumps(msg, ensure_ascii=False) + '\n')
                writers[variant].flush()
            labels = ' '.join(f'{v}={"good" if preds[v][0] else "bad"}' for v in VARIANTS)
            print(f'REP {published} [{event.start_frame},{event.end_frame}] {labels}', flush=True)

        try:
            while args.max_frames == 0 or n < args.max_frames:
                okay, frame = cap.read()
                if not okay:
                    break
                fid = args.start_frame + n
                res = pose_model.predict(frame, device=args.device, imgsz=640,
                                         conf=.25, verbose=False)[0]
                kp = absent()
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    idx = select_target(boxes, scores, anchor, previous_box)
                    if idx is not None and res.keypoints is not None:
                        kp = res.keypoints.data[idx].cpu().numpy().astype(np.float32)
                        kp[kp[:, 2] < .3, :2] = np.nan
                        previous_box = boxes[idx].copy()
                        chosen += 1
                completed = counter.push(kp, frame_id=fid)
                for event in buffer.push(fid, kp, completed):
                    emit(event, fid)
                n += 1
                if n % 150 == 0:
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    print(f'frame={fid} state={counter.state} reps={published} '
                          f'pose_found={chosen}/{n} processing_fps={n/elapsed:.1f}',flush=True)
                if args.paced:
                    remaining = n/fps - (time.perf_counter() - t0)
                    if remaining > 0:
                        time.sleep(remaining)
            for event in buffer.flush():
                emit(event, args.start_frame + n - 1)
        finally:
            cap.release()

    elapsed = max(time.perf_counter() - t0, 1e-6)
    print(f'DONE: video={args.video} frames={n} reps={published} '
          f'pose_found={chosen}/{n or 1} processing_fps={n/elapsed:.1f}')
    print(f'Saved three synchronized event files under {args.output_root}')
    print('GT labels and boundaries were not used during streaming inference.')


if __name__ == '__main__':
    main()
