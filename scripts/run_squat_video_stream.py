"""Online YOLOv8m-Pose -> tracking -> repetition segmentation -> LR classification.

The source is a complete MP4, NOT pre-cut NPZ repetitions. No Segmentation.csv
is used to determine repetitions. This is a WSL prototype, not a realtime FPS
performance claim. Requires a manually selected anchor per dataset recording.
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
from train_squat_extended import extract_feature_groups
from stream_squat_core import SquatCounter


def absent():
    out = np.zeros((17, 3), np.float32)
    out[:, :2] = np.nan
    return out


def classify(seq, artifact):
    features = extract_feature_groups(seq)
    record = {}
    record.update(features['base'])
    record.update(features['asymmetry'])
    order = artifact['feature_names']
    if len(order) != 180 or set(order) != set(record):
        raise RuntimeError('Feature schema does not match PlusAsymmetry180')
    values = np.array([[record[name] for name in order]], dtype=float)
    model = artifact['pipeline']
    pred = int(model.predict(values)[0])
    correct_idx = list(model.classes_).index(1) if hasattr(model, 'classes_') else None
    if correct_idx is None:
        # sklearn Pipeline exposes classes_ on modern versions, but fall back
        # to the final estimator for older versions.
        correct_idx = list(model.named_steps['model'].classes_).index(1)
    prob = float(model.predict_proba(values)[0][correct_idx])
    return pred, prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', default='PM_022')
    p.add_argument('--video-file', type=Path, default=None)
    p.add_argument('--anchor-config', type=Path,
                   default=Path('configs/rehab24_squat_camera17_subjects.json'))
    p.add_argument('--model', type=Path, default=Path('models/squat_plus_asymmetry180.joblib'))
    p.add_argument('--yolo', default='yolov8m-pose.pt')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=1200)
    p.add_argument('--device', default='0')
    p.add_argument('--output', type=Path, default=Path('results/stream_test/PM_022_events.jsonl'))
    p.add_argument('--paced', action='store_true', help='Simulate video playback speed, if inference is fast enough')
    p.add_argument('--preview', action='store_true', help='Display preview with cv2.imshow')
    args = p.parse_args()
    video = args.video_file or Path(f'data/raw/REHAB24-6/{args.video}-Camera17-30fps.mp4')
    if not video.is_file():
        raise FileNotFoundError(video)
    anchor_cfg = json.loads(args.anchor_config.read_text())
    if args.video not in anchor_cfg:
        raise ValueError(f'Select anchor for {args.video} first (select_subjects.py)')
    anchor = np.array(anchor_cfg[args.video]['bbox'], dtype=np.float32)
    artifact = joblib.load(args.model)  # Load only self-created trusted artifacts.
    if artifact.get('feature_variant') != 'PlusAsymmetry180':
        raise ValueError('Expected PlusAsymmetry180 artifact')
    model = YOLO(args.yolo)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video: ' + str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    detector = SquatCounter(fps=fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n, chosen, count = 0, 0, 0
    prev = None
    with args.output.open('w', encoding='utf-8') as out:
        while args.max_frames <= 0 or n < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            fid = args.start_frame + n
            res = model.predict(frame, device=args.device, imgsz=640, conf=.25, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                kp = absent()
            else:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                i = select_target(boxes, scores, anchor, prev)
                if i is not None and res.keypoints is not None:
                    kp = res.keypoints.data[i].cpu().numpy().astype(np.float32)
                    kp[kp[:, 2] < .3, :2] = np.nan
                    prev = boxes[i].copy()
                    chosen += 1
                else:
                    kp = absent()
            event = detector.push(kp, frame_id=fid)
            if event is not None:
                try:
                    pred, prob_correct = classify(event.keypoints, artifact)
                    count += 1
                    message = {
                        'type': 'rep', 'exercise': 'squat', 'count': count,
                        'posture': 'good' if pred == 1 else 'bad',
                        'issues': [], 'score': None,
                        'probability_correct_uncalibrated': round(prob_correct, 4),
                        'start_frame': event.start_frame, 'end_frame': event.end_frame,
                        'peak_hip_depth_torso_units': round(event.peak_depth, 3),
                        'min_projected_knee_angle': round(event.min_knee_angle, 1),
                    }
                except Exception as exc:
                    # Count completed reps even when a classifier feature fails.
                    count += 1
                    message = {'type':'rep','exercise':'squat','count':count,
                               'posture':'unknown','issues':[],'score':None,
                               'start_frame':event.start_frame,'end_frame':event.end_frame,
                               'classification_error':str(exc)}
                out.write(json.dumps(message, ensure_ascii=False) + '\n')
                out.flush()
                print('REP', json.dumps(message, ensure_ascii=False), flush=True)
            n += 1
            if n % 150 == 0:
                elapsed = max(time.perf_counter()-t0, 1e-6)
                print(f'frame={fid} state={detector.state} reps={count} pose_found={chosen}/{n} processing_fps={n/elapsed:.1f}', flush=True)
            if args.preview:
                cv2.putText(frame, f'{detector.state} reps={count}', (25, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.imshow('Squat stream test (press q)', cv2.resize(frame, (960, 540)))
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
            if args.paced:
                # Playback pacing, NOT a guarantee of 30 FPS inference.
                remaining = (n/fps) - (time.perf_counter()-t0)
                if remaining > 0:
                    time.sleep(remaining)
    cap.release()
    if args.preview:
        cv2.destroyAllWindows()
    elapsed = max(time.perf_counter()-t0, 1e-6)
    print(f'DONE: {n} frames, {count} complete reps, pose_found={chosen}/{n if n else 1}, '
          f'wall={elapsed:.1f}s, actual_processing_fps={n/elapsed:.1f}, output={args.output}')
    print('No ground truth used in online detection. Compare to Segmentation.csv separately.')


if __name__ == '__main__':
    main()
