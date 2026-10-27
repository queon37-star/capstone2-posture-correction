"""Pose-independent feature extraction and squat Good/Bad classification.

Input: a complete repetition as [T, 17, 3] COCO keypoints in source pixels.
Does not import Ultralytics or invoke a Pose model.
"""

import numpy as np
from train_squat_extended import extract_feature_groups


def absent():
    """Return one missing COCO17 observation with zero confidences."""
    out = np.zeros((17, 3), np.float32)
    out[:, :2] = np.nan
    return out


def classify(seq, artifact):
    """Apply the existing fitted 180-feature sklearn pipeline (no retraining)."""
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
