"""Controlled bottom-depth threshold experiment for buffered squat stream.

Only change from run_squat_video_stream_buffered.py: optional command-line
bottom_depth / bottom_knee passed to the existing SquatCounter. Defaults
match original. GT files are NEVER loaded by this script.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

from extract_all_squats_tracked import select_target
from run_squat_video_stream import classify, absent
from stream_squat_core import SquatCounter
from squat_rep_buffer import RepBoundaryBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='PM_043')
    parser.add_argument('--video-file', type=Path, default=None)
    parser.add_argument('--anchor-config', type=Path,
                        default=Path('configs/rehab24_squat_camera17_subjects.json'))
    parser.add_argument('--model', type=Path, default=Path('models/squat_plus_asymmetry180.joblib'))
    parser.add_argument('--yolo', default='yolov8m-pose.pt')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--device', default='0')
    parser.add_argument('--pre-frames', type=int, default=16)
    parser.add_argument('--post-frames', type=int, default=22)
    parser.add_argument('--bottom-depth', type=float, default=0.45,
                        help='Hip displacement threshold in torso units (default: original 0.45)')
    parser.add_argument('--bottom-knee', type=float, default=120.0,
                        help='Projected knee angle bottom threshold (default: original 120 degrees)')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--paced', action='store_true')
    args = parser.parse_args()

    if args.start_frame < 0:
        parser.error('--start-frame must be >= 0')
    if not 0.16 < args.bottom_depth < 2.0:
        parser.error('--bottom-depth must be greater than start_depth=0.16 and less than 2.0')
    if not 0 < args.bottom_knee <= 180:
        parser.error('--bottom-knee must be in (0, 180]')
    video_path = args.video_file or Path(f'data/raw/REHAB24-6/{args.video}-Camera17-30fps.mp4')
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    config = json.loads(args.anchor_config.read_text(encoding='utf-8'))
    if args.video not in config:
        raise ValueError(f'Missing subject anchor for {args.video}')
    anchor = np.asarray(config[args.video]['bbox'], np.float32)
    artifact = joblib.load(args.model)  # Only load own trusted model file.
    if artifact.get('feature_variant') != 'PlusAsymmetry180':
        raise ValueError('Expected PlusAsymmetry180 classifier artifact')
    pose_model = YOLO(args.yolo)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    counter = SquatCounter(fps=fps, bottom_depth=args.bottom_depth,
                           bottom_knee=args.bottom_knee)
    print(f'Counter thresholds: bottom_depth={counter.bottom_depth:.3f} torso units, '
          f'bottom_knee={counter.bottom_knee:.1f} deg; other state machine rules unchanged.', flush=True)
    assembler = RepBoundaryBuffer(fps=fps, pre_frames=args.pre_frames, post_frames=args.post_frames)
    out_path = args.output or Path(f'results/stream_test/{args.video}_buffered_events.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    n = pose_found = 0
    previous_box = None
    event_count = 0

    def publish(completed_rep, writer):
        nonlocal event_count
        event_count += 1
        try:
            pred, prob = classify(completed_rep.keypoints, artifact)
            message = {
                'type': 'rep', 'exercise': 'squat', 'count': event_count,
                'posture': 'good' if pred == 1 else 'bad',
                'issues': [], 'score': None,
                'probability_correct_uncalibrated': round(prob, 4),
            }
        except Exception as exc:
            message = {'type': 'rep', 'exercise': 'squat', 'count': event_count,
                       'posture': 'unknown', 'issues': [], 'score': None,
                       'classification_error': str(exc)}
        message.update({
            'start_frame': completed_rep.start_frame,
            'end_frame': completed_rep.end_frame,
            'detected_start_frame': completed_rep.detected_start_frame,
            'detected_end_frame': completed_rep.detected_end_frame,
            'classification_frame_count': len(completed_rep.keypoints),
            'post_context_complete': completed_rep.post_context_complete,
            'peak_hip_depth_torso_units': round(completed_rep.peak_depth, 3),
            'min_projected_knee_angle': round(completed_rep.min_knee_angle, 1),
        })
        writer.write(json.dumps(message, ensure_ascii=False) + '\n')
        writer.flush()
        print('REP', json.dumps(message, ensure_ascii=False), flush=True)

    with out_path.open('w', encoding='utf-8') as output:
        while args.max_frames <= 0 or n < args.max_frames:
            success, frame = cap.read()
            if not success:
                break
            fid = args.start_frame + n
            result = pose_model.predict(frame, device=args.device,
                                        imgsz=640, conf=.25, verbose=False)[0]
            kp = absent()
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                idx = select_target(boxes, scores, anchor, previous_box)
                if idx is not None and result.keypoints is not None:
                    kp = result.keypoints.data[idx].cpu().numpy().astype(np.float32)
                    kp[kp[:, 2] < .3, :2] = np.nan
                    previous_box = boxes[idx].copy()
                    pose_found += 1
            rep_event = counter.push(kp, frame_id=fid)
            for rep in assembler.push(fid, kp, rep_event):
                publish(rep, output)
            n += 1
            if n % 150 == 0:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                print(f'frame={fid} state={counter.state} detected_reps={counter.count} '
                      f'published_reps={event_count} pose_found={pose_found}/{n} '
                      f'processing_fps={n/elapsed:.1f}', flush=True)
            if args.paced:
                remaining = n/fps - (time.perf_counter() - start_time)
                if remaining > 0:
                    time.sleep(remaining)
        for rep in assembler.flush():
            publish(rep, output)
    cap.release()
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    print(f'DONE: {n} frames, {event_count} complete reps, pose_found={pose_found}/{n or 1}, '
          f'wall={elapsed:.1f}s, actual_processing_fps={n/elapsed:.1f}, output={out_path}')
    print(f'Boundary padding: -{args.pre_frames} frames / +{args.post_frames} frames; '
          'pending post-context introduces bounded classification delay.')
    print('No ground truth used during streaming detection or classification.')


if __name__ == '__main__':
    main()
