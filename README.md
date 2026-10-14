# Capstone2 - Exercise Posture Correction System

## 1. Project Overview

This repository contains the AI posture-analysis prototype for the
Capstone2 exercise posture correction system.

The current pipeline is:

1. YOLOv8m-Pose: Extract 17 human keypoints.
2. Subject Tracking: Select and track the exercise participant.
3. SquatCounter: Detect squat repetitions using a state machine.
4. RepBoundaryBuffer: Expand detected repetition boundaries.
5. Feature Extraction: Extract 180 geometric and temporal features.
6. Logistic Regression: Classify each repetition as Good or Bad.

## 2. Latest Inference Configuration

- Model: `models/squat_plus_asymmetry180_crop10_all9.joblib`
- Feature variant: `PlusAsymmetry180`
- Feature dimensions: 180
- Training method: 10% temporal-crop augmentation
- Squat bottom-depth threshold: 0.20
- Pre-buffer: 16 frames
- Post-buffer: 22 frames

The latest execution wrapper is:

`bash scripts/run_squat_latest.sh --help`

The wrapper passes the validated configuration to
`scripts/run_squat_video_stream_tunable.py`.

## 3. Evaluation Results

Dataset: REHAB24-6, Exercise 6 (Squat), Camera17, 30 FPS.

- Subjects: 9
- Ground-truth repetitions: 195
- Good repetitions: 134
- Bad repetitions: 61

### Repetition Detection

- Detected repetitions: 195/195
- Mean interval IoU: 0.860

### Good/Bad Classification

Evaluated using Leave-One-Person-Out (LOPO) cross-validation.

- Accuracy: 168/195 (86.2%)
- Bad recall: 54/61 (88.5%)
- Bad precision: 54/74 (73.0%)
- Bad F1 score: approximately 80.0%

These classification metrics are from held-out-subject
evaluation, not evaluation of the final all-subject model
on its own training data.

### Processing Performance

- PC processing throughput: 83.2 FPS
- Evaluation input: recorded video
- Development GPU: NVIDIA RTX 4070 Ti

This FPS is not the measured performance of the
reComputer AI R2130-12.

## 4. Current Development Status

Completed:
- YOLOv8m-Pose-based keypoint extraction on PC
- Anchor-based subject tracking
- Streaming squat repetition detection
- Repetition boundary buffering
- 180-feature Good/Bad classification
- LOPO evaluation and model comparison
- Hailo-8 pose inference environment setup on R2130-12

Remaining:
- Integrate the complete AI pipeline on R2130-12
- Test with live camera input
- Measure end-to-end FPS, latency and memory usage
- Validate counting and classification on unseen recordings
- Integrate real AI events with the server and mobile app

## 5. Repository Notes

Original datasets, videos, generated keypoints, evaluation outputs
and logs are excluded from Git.

The final Crop10 model was fitted using all nine subjects.
The LOPO results were obtained using separate training/test folds.

Only load trusted, self-generated Joblib model files.

## 6. Original Baseline Documentation

### REHAB24-6 YOLOv8m-Pose Squat classification baseline

Inputs: `data/processed/REHAB24-6/camera17_yolov8m_tracked/index.csv` and its 195 `PM_*_rep*.npz` files. Each `.npz` stores `(T, 17, 3)` landmarks, `label` (`1=correct`, `0=incorrect`), and `person_id`.

From your WSL project root:

```bash
conda activate capstone_pose
cp train_squat_baseline.py scripts/
python scripts/train_squat_baseline.py 2>&1 | tee logs/train_squat_baseline.log
```

`results/squat_baseline/` will contain `metrics.json`, `confusion_matrix.png`, per-subject scores, `lopo_predictions.csv`, `squat_classifier.joblib`, and preprocessing features.

This is a **repetition-level, offline** classifier and not a live per-frame form judgment. It uses geometric features (knee/hip angles, torso tilt, hip displacement) from 2D YOLO landmarks, then compares RandomForest and LogisticRegression under Leave-One-Person-Out validation. **No subject ID, label, camera orientation or video ID is used as a feature.** It selects the final model based on the held-out subject performance and then fits the final pipeline on all available participants. Since the same CV scores were used to compare models, consider the selected score exploratory rather than an unbiased final model selection estimate.

YOLO valid joint proportion and target selection are separate quality concerns; manually inspect representative videos and samples. The model cannot yet generate clinically valid named error diagnoses (`issues`) or calibrated `score` values; these require separately labeled data/validation.
