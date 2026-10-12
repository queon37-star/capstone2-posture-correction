"""Choose the exercise participant once per recording for fixed-camera REHAB24-6.

Generates an annotated JPG and asks for the bounding-box index. No GUI Python
backend is required: on WSL it tries to open the JPG in Windows automatically.
"""
import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path('data/raw/REHAB24-6')
CONFIG = Path('configs/rehab24_squat_camera17_subjects.json')
PREVIEWS = Path('results/subject_selection')


def show_in_windows(path):
    try:
        win_path = subprocess.check_output(['wslpath', '-w', str(path.resolve())], text=True).strip()
        subprocess.Popen(['explorer.exe', win_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', help='e.g. PM_022 (default: all squat recordings)')
    ap.add_argument('--force', action='store_true', help='reselect an existing anchor')
    ap.add_argument('--no-open', action='store_true', help='do not launch Windows image viewer')
    args = ap.parse_args()

    with (BASE / 'Segmentation.csv').open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    by_video = defaultdict(list)
    for row in rows:
        if row['exercise_id'] == '6':
            by_video[row['video_id']].append(row)
    videos = [args.video] if args.video else sorted(by_video)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    anchors = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    model = YOLO('yolov8m-pose.pt')

    for video_id in videos:
        if video_id not in by_video:
            raise ValueError(f'Unknown squat video ID: {video_id}')
        if video_id in anchors and not args.force:
            print(f'{video_id}: anchor already selected (use --force to change)')
            continue
        first = sorted(by_video[video_id], key=lambda r: int(r['first_frame']))[0]
        idx = (int(first['first_frame']) + int(first['last_frame'])) // 2
        video_path = BASE / f'{video_id}-Camera17-30fps.mp4'
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {video_path}')
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'Cannot read {video_id} frame={idx}')

        result = model.predict(frame, device=0, imgsz=640, conf=0.25, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            raise RuntimeError(f'No people detected in {video_id} frame={idx}; select another frame')
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        preview = frame.copy()
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 255), 5)
            cv2.putText(preview, f'ID {i} / {score:.2f}', (x1, max(40, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
        preview_path = PREVIEWS / f'{video_id}_choose.jpg'
        cv2.imwrite(str(preview_path), preview)
        print(f'\n{video_id}: first repetition midpoint frame {idx}, people={len(boxes)}')
        print(f'Image with candidate IDs: {preview_path.resolve()}')
        if not args.no_open:
            show_in_windows(preview_path)
        while True:
            answer = input(f'Choose the exercise subject ID (0..{len(boxes)-1}, q=abort): ').strip()
            if answer.lower() == 'q':
                print('Cancelled; existing selections were saved.')
                return
            if answer.isdecimal() and 0 <= int(answer) < len(boxes):
                selected = int(answer)
                break
            print('Invalid selection; check the JPG labels.')
        anchors[video_id] = {
            'bbox': [float(v) for v in boxes[selected]],
            'anchor_frame': idx,
            'frame_width': int(frame.shape[1]),
            'frame_height': int(frame.shape[0]),
            'selected_id': selected,
        }
        CONFIG.write_text(json.dumps(anchors, indent=2, ensure_ascii=False) + '\n')
        print(f'Saved anchor for {video_id} -> {CONFIG}')

    print(f'\nDone. Anchors: {len(anchors)}; file: {CONFIG}')


if __name__ == '__main__':
    main()
