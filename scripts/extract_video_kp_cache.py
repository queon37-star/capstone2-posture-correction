"""Full-video COCO17 keypoint cache for offline experiments (PC / WSL only).

Runs YOLOv8-Pose ONCE per video frame and records the target person's raw
keypoints for EVERY decoded frame, including frames where nothing was selected.
Later experiments (gating, buffer policy, classifiers) replay the cache instead
of re-running YOLO, so every condition sees identical inputs.

Target selection is the existing fixed-ROI + temporal rule
(extract_all_squats_tracked.select_target + configs/rehab24_squat_camera17_subjects.json),
applied frame by frame exactly as in run_squat_stream_model_comparison.py.
Keypoints are stored RAW (no confidence masking); the replay tool applies the
stream pipeline's `conf < 0.3 -> NaN` masking so the choice stays explicit.

Outputs per video, in --out-dir (nothing else is written or modified):
  <video>.kp.bin         228-byte records, same layout as SQUAT_KP_DUMP
                         (frame_id, t=frame_id/fps, pose_valid, kp[17,3])
  <video>.kp.bin.json    metadata: video identity/hash, fps, YOLO settings, status counts
  <video>.frames.npz     per-frame status, person count, selected box/score

  python scripts/extract_video_kp_cache.py --all
  python scripts/extract_video_kp_cache.py --video PM_029 --device 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_detect_hailo_bridge import KP_DUMP_DTYPE  # noqa: E402  (one record layout everywhere)

CACHE_FORMAT = 'capstone2-kp-cache'
CACHE_VERSION = 1
DEFAULT_VIDEOS = ['PM_008', 'PM_022', 'PM_029', 'PM_038', 'PM_043',
                  'PM_105', 'PM_113', 'PM_118', 'PM_126']

# Per-frame status codes stored in <video>.frames.npz
STATUS_SELECTED = 0          # target chosen, keypoints stored
STATUS_NO_DETECTION = 1      # YOLO returned no person box
STATUS_NO_TARGET = 2         # boxes exist but none passes ROI/continuity rule
STATUS_NO_KEYPOINTS = 3      # target box chosen but the result had no keypoints
STATUS_NAMES = {STATUS_SELECTED: 'selected', STATUS_NO_DETECTION: 'no_detection',
                STATUS_NO_TARGET: 'no_target', STATUS_NO_KEYPOINTS: 'no_keypoints'}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def git_commit(repo):
    try:
        out = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=5)
        return {'commit': out.stdout.strip() or None, 'tracked_changes': bool(dirty.stdout.strip())}
    except Exception:  # git missing is not an error for extraction
        return {'commit': None, 'tracked_changes': None}


class UltralyticsPoseBackend:
    """predict(frame) -> (boxes[N,4] xyxy, scores[N], keypoints[N,17,3] or None)."""

    def __init__(self, weights, *, device, imgsz, conf):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.weights, self.device, self.imgsz, self.conf = weights, device, imgsz, conf

    def predict(self, frame):
        res = self.model.predict(frame, device=self.device, imgsz=self.imgsz,
                                 conf=self.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), None
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        kps = None if res.keypoints is None else res.keypoints.data.cpu().numpy().astype(np.float32)
        return boxes, scores, kps

    def describe(self):
        info = {'backend': 'ultralytics', 'weights': str(self.weights), 'device': str(self.device),
                'imgsz': self.imgsz, 'conf': self.conf, 'call': 'model.predict(frame, ...)[0] per frame'}
        try:
            import ultralytics
            info['ultralytics_version'] = ultralytics.__version__
            import torch
            info['torch_version'] = torch.__version__
            info['cuda_device'] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        except Exception as exc:  # metadata only
            info['version_error'] = repr(exc)
        ckpt = Path(str(self.weights))
        if ckpt.is_file():
            info['weights_sha256'] = sha256_file(ckpt)
            info['weights_path_resolved'] = str(ckpt.resolve())
        return info


def missing_kp():
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, :2] = np.nan
    return kp


def extract_video(video_id, video_path, anchor_entry, backend, out_dir, *,
                  anchor_config_path=None, max_frames=0, overwrite=False, progress_every=300):
    """Write <video_id>.kp.bin/.json/.frames.npz. Returns the metadata dict."""
    import cv2
    from extract_all_squats_tracked import select_target  # existing rule, unchanged

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_bin = out_dir / f'{video_id}.kp.bin'
    final_meta = out_dir / f'{video_id}.kp.bin.json'
    final_frames = out_dir / f'{video_id}.frames.npz'
    if not overwrite and any(p.exists() for p in (final_bin, final_meta, final_frames)):
        raise FileExistsError(f'{video_id}: cache already exists in {out_dir} (use --overwrite)')

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f'{video_path}: FPS unavailable')
    anchor = np.asarray(anchor_entry['bbox'], dtype=np.float32)

    meta = {
        'format': CACHE_FORMAT, 'version': CACHE_VERSION, 'complete': False,
        'source': 'pc_ultralytics_cache',
        'record_bytes': KP_DUMP_DTYPE.itemsize,
        'assumed_fps': fps, 'timebase': 'file',          # read by replay_kp_dump.py
        'keypoints': 'raw model output for the selected person (NOT confidence-masked)',
        'recommended_replay_mask_conf': 0.30,
        'video': {'video_id': video_id, 'path': str(video_path), 'resolved': str(video_path.resolve()),
                  'size_bytes': video_path.stat().st_size, 'sha256': sha256_file(video_path),
                  'fps': fps, 'frame_count_reported': reported, 'width': width, 'height': height},
        'pose_model': backend.describe(),
        'target_selection': {
            'rule': 'extract_all_squats_tracked.select_target(boxes, scores, anchor, previous_box)',
            'previous_box_update': 'only when a target is selected (as in run_squat_stream_model_comparison.py)',
            'anchor_bbox': [float(v) for v in anchor], 'anchor_frame': anchor_entry.get('anchor_frame'),
            'anchor_config': str(anchor_config_path) if anchor_config_path else None,
            'anchor_config_sha256': (sha256_file(anchor_config_path)
                                     if anchor_config_path and Path(anchor_config_path).is_file() else None)},
        'status_codes': {str(k): v for k, v in STATUS_NAMES.items()},
        'code': git_commit(Path(__file__).resolve().parents[1]),
        'host': {'python': sys.version.split()[0], 'platform': platform.platform()},
        'max_frames': max_frames,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
    }

    tmp_bin = final_bin.with_name(final_bin.name + '.partial')
    record = np.zeros(1, dtype=KP_DUMP_DTYPE)
    statuses, persons, boxes_out, scores_out = [], [], [], []
    previous_box = None
    n = 0
    t0 = time.perf_counter()
    try:
        with tmp_bin.open('wb') as f:
            while max_frames <= 0 or n < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                boxes, scores, kps = backend.predict(frame)
                kp, status, box, score = missing_kp(), STATUS_NO_DETECTION, None, np.nan
                if len(boxes):
                    idx = select_target(boxes, scores, anchor, previous_box)
                    if idx is None:
                        status = STATUS_NO_TARGET
                    elif kps is None:
                        status = STATUS_NO_KEYPOINTS
                        box, score = boxes[idx], float(scores[idx])
                        # stream loop updates previous_box only with keypoints: keep that
                    else:
                        status = STATUS_SELECTED
                        kp = kps[idx].astype(np.float32, copy=True)
                        box, score = boxes[idx], float(scores[idx])
                        previous_box = boxes[idx].copy()
                record['frame_id'] = n
                record['t'] = n / fps
                record['pose_valid'] = 1 if status == STATUS_SELECTED else 0
                record['kp'][0] = kp
                f.write(record.tobytes())
                statuses.append(status)
                persons.append(len(boxes))
                boxes_out.append(np.full(4, np.nan, np.float32) if box is None else np.asarray(box, np.float32))
                scores_out.append(score)
                n += 1
                if progress_every and n % progress_every == 0:
                    el = max(time.perf_counter() - t0, 1e-6)
                    print(f'{video_id}: frame {n}/{reported or "?"} selected='
                          f'{statuses.count(STATUS_SELECTED)} fps={n / el:.1f}', flush=True)
    finally:
        cap.release()

    elapsed = time.perf_counter() - t0
    status_arr = np.asarray(statuses, dtype=np.uint8)
    np.savez_compressed(final_frames.with_name(final_frames.name + '.partial.npz'),
                        frame_id=np.arange(n, dtype=np.int64), status=status_arr,
                        person_count=np.asarray(persons, dtype=np.int16),
                        selected_box=np.asarray(boxes_out, dtype=np.float32).reshape(n, 4),
                        selected_score=np.asarray(scores_out, dtype=np.float32))
    meta.update({
        'complete': max_frames <= 0 and (reported == 0 or n == reported),
        'frames_decoded': n,
        'decode_stopped_early': bool(reported and n < reported and max_frames <= 0),
        'status_counts': {STATUS_NAMES[k]: int((status_arr == k).sum()) for k in STATUS_NAMES},
        'multi_person_frames': int((np.asarray(persons) > 1).sum()),
        'extraction_seconds': round(elapsed, 2),
        'extraction_fps': round(n / max(elapsed, 1e-6), 2),
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    os.replace(tmp_bin, final_bin)
    os.replace(final_frames.with_name(final_frames.name + '.partial.npz'), final_frames)
    final_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{video_id}: {n} frames -> {final_bin} complete={meta["complete"]} '
          f'status={meta["status_counts"]}', flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', action='append', help='video id, repeatable (e.g. PM_029)')
    ap.add_argument('--all', action='store_true', help='the nine Ex6 Camera17 videos')
    ap.add_argument('--video-dir', type=Path, default=Path('data/raw/REHAB24-6'))
    ap.add_argument('--video-file', type=Path, help='explicit file (only with one --video)')
    ap.add_argument('--anchor-config', type=Path, default=Path('configs/rehab24_squat_camera17_subjects.json'))
    ap.add_argument('--out-dir', type=Path, default=Path('data/cache/kp_full/rehab24_camera17_yolov8m'))
    ap.add_argument('--yolo', default='yolov8m-pose.pt')
    ap.add_argument('--device', default='0')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--max-frames', type=int, default=0, help='debug only; result is marked incomplete')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    videos = DEFAULT_VIDEOS if args.all else (args.video or [])
    if not videos:
        ap.error('give --video ID or --all')
    if args.video_file and len(videos) != 1:
        ap.error('--video-file needs exactly one --video')
    anchors = json.loads(args.anchor_config.read_text(encoding='utf-8'))
    missing = [v for v in videos if v not in anchors]
    if missing:
        raise KeyError(f'no anchor for {missing} in {args.anchor_config}')
    backend = UltralyticsPoseBackend(args.yolo, device=args.device, imgsz=args.imgsz, conf=args.conf)
    for v in videos:
        path = args.video_file or args.video_dir / f'{v}-Camera17-30fps.mp4'
        extract_video(v, path, anchors[v], backend, args.out_dir,
                      anchor_config_path=args.anchor_config, max_frames=args.max_frames,
                      overwrite=args.overwrite)


if __name__ == '__main__':
    main()
