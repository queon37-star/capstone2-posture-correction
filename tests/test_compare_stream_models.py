"""Small unit tests of aligned feature/class encoding; no GPU/video required."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_squat_stream_model_comparison import classify_three


class ToyPipeline:
    classes_ = np.array([0, 1])
    def __init__(self, threshold):
        self.threshold = threshold
    def predict_proba(self, x):
        p = 1 / (1 + np.exp(-(x[:, 0] - self.threshold)))
        return np.column_stack([1-p, p])
    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def test_feature_order_and_common_sequence():
    feats = {'base': {f'f{i}': float(i) for i in range(160)},
             'asymmetry': {f'f{i}': float(i) for i in range(160, 180)}}
    names = list(feats['base']) + list(feats['asymmetry'])
    artifacts = {
        'one': {'feature_names': names, 'pipeline': ToyPipeline(-1)},
        'two': {'feature_names': names[::-1], 'pipeline': ToyPipeline(178)},
        'three': {'feature_names': names, 'pipeline': ToyPipeline(1)},
    }
    stream = np.ones((30, 17, 3), dtype=np.float32)
    called = []
    def extract(seq):
        called.append(id(seq))
        assert seq is stream
        return feats
    out = classify_three(stream, artifacts, extract)
    assert len(called) == 1
    assert out['one'][0] == 1
    assert out['two'][0] == 1
    assert out['three'][0] == 0
    for _, prob in out.values():
        assert 0 <= prob <= 1


def test_schema_check():
    try:
        classify_three(np.zeros((20, 17, 3)), {'bad': {
            'feature_names': ['nonexistent'], 'pipeline': ToyPipeline(0),
        }}, lambda seq: {'base': {'f1': 0}, 'asymmetry': {}})
    except ValueError as exc:
        assert 'feature schema mismatch' in str(exc)
    else:
        raise AssertionError('Expected schema exception')


if __name__ == '__main__':
    test_feature_order_and_common_sequence()
    test_schema_check()
    print('Three-model shared-keypoint inference tests: PASS (synthetic only)')
