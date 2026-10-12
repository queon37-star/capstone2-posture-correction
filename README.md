# REHAB24-6 YOLOv8m-Pose Squat classification baseline

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
