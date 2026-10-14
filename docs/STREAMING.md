# Squat Streaming - Current Status

## 1. Latest Configuration

The original streaming prototype has been extended with
improved repetition detection and classification.

Current validated WSL configuration:

- Bottom-depth threshold: 0.20
- Pre-buffer: 16 frames
- Post-buffer: 22 frames
- Features: PlusAsymmetry180
- Classifier: Logistic Regression
- Training augmentation: 10% temporal crop
- Final model: `models/squat_plus_asymmetry180_crop10_all9.joblib`

## 2. Execution

Run the latest configuration using:

`bash scripts/run_squat_latest.sh --help`

This wrapper uses the tunable streaming implementation
with the latest model and threshold settings.

The input is an uncut recorded video.

The system performs pose inference, target tracking,
repetition detection and Good/Bad classification.

Ground-truth repetition boundaries are not used
during streaming inference.

## 3. Validation

REHAB24-6 Camera17 evaluation results:

- Repetition detection: 195/195
- Mean interval IoU: 0.860
- LOPO classification accuracy: 86.2%
- Bad recall: 88.5%

The evaluated system is a recorded-video streaming prototype.

Real-camera end-to-end validation on the
reComputer AI R2130-12 is still pending.

## 4. Earlier Prototype Documentation

### Capstone2: Squat online posture prototype (WSL)

This directory adds a provisional **streaming repetition detector** to the existing pose-training project. It is an engineering prototype, **NOT yet validated online detection performance**.

## What the old model does

The trained Logistic Regression uses a variable-length sequence of one *complete squat repetition*, `[T, 17, 3]` 2D YOLO keypoints. A sequence is converted to the fixed-sized 180-feature vector. Training/evaluation originally used repetition boundaries from REHAB24-6 `Segmentation.csv`.

## What the new prototype does

Uncut original MP4 -> YOLOv8m-Pose -> existing anchor/ROI person tracking -> stateful detector -> completed repetition -> `base + asymmetry` features -> Logistic Regression -> JSON Lines. **No segmentation annotations enter the online pipeline.**

The squat detector is hysteresis threshold-based and needs calibration at initially standing pose. It uses hip vertical displacement normalized by torso and projected knee angle. It may fail with camera changes, occlusions, atypical motion, or view-dependent 2D angles. Its thresholds are provisional, not yet tuned to this dataset.

## Installation (run from existing WSL project root)

Copy the three Python scripts into `scripts/`. The scripts use your existing `train_squat_baseline.py`, `train_squat_extended.py` and `extract_all_squats_tracked.py`.

```bash
python scripts/train_plus_asymmetry180.py
python tests/test_stream_squat_core.py
python scripts/run_squat_video_stream.py --video PM_022 --max-frames 1200 \
    2>&1 | tee logs/stream_pm022.log
```

Requires same `capstone_pose` Python environment (numpy, pandas, sklearn, OpenCV, Ultralytics, joblib, CUDA-compatible torch). `models/squat_plus_asymmetry180.joblib` is a complete-data refit, distinct from the out-of-person LOPO predictions. **Only load your own trusted joblib artifacts.**

CLI options include `--max-frames 0` (whole video), `--preview` (requires GUI), `--paced` (real-time-like playback if processing is faster), `--start-frame`, `--device` and `--output`. `actual_processing_fps` indicates the observed speed, while `--paced` never makes inference faster. First execution should use at most 1200 frames to evaluate thresholds and timing.

### Output

`results/stream_test/PM_022_events.jsonl`. Each completed repetition has `count`, `posture: good|bad|unknown`, `issues: []`, `score: null`, `start_frame/end_frame`. The `probability_correct_uncalibrated` is a model probability and MUST NOT be treated as a calibrated posture score; no detailed fault types can be produced by a binary classifier.

## Validation before device port

1. Use an *uncut* MP4 and independently compare emitted start/end frames and rep count to `Segmentation.csv` (post-hoc only). Tune thresholds on development videos; reserve unseen videos/subjects for later external testing.
2. Measure actual processing FPS; this WSL prototype may run more slowly than the source 30fps. Streaming architecture and achieving real-time speed are separate requirements.
3. Check a variety of persons, side/front view, false reps, incomplete motions, and classifier performance with **predicted** boundaries rather than ground-truth boundaries. Do not cite earlier 86.67% LOPO as the accuracy of this end-to-end pipeline.
4. When moving to reComputer/Hailo, replace only the pose input adapter while preserving the same state machine and classifier feature schema.

## GitHub

Commit code, README, dependency manifests, and small self-generated models. Exclude source videos, segmentation CSV, derived keypoints and logs (see `.gitignore`). License and permissions for REHAB24-6 data must be honored. GitHub push is performed from the user's WSL machine, as that contains the actual working project.

### Independent post-hoc annotation comparison

For a 0..1199 frame replay, **after the streaming script finishes**, compare predicted boundaries with the annotation file:

```bash
python scripts/evaluate_squat_stream.py --video PM_022 --start-frame 0 --end-frame 1199
```

The script uses `Segmentation.csv` only **after the fact** and reports matched/missed/extra reps, interval IoU and boundary MAE. A partial ground-truth repetition at a window edge is excluded from the comparison.
