# Generated from frozen offline source by prepare_board_runtime.py.
from __future__ import annotations
import numpy as np
from .squat_features import extract_feature_groups

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
