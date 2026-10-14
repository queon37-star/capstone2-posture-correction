"""Synthetic-only check for full/trim start/trim end/trim both training rows."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_squat_crop10_all9 import make_augmented_rows


def fake_extractor(seq):
    return {'base': {'duration': float(len(seq)), 'first': float(seq[0,0,0])},
            'asymmetry': {'last': float(seq[-1,0,0])}}


def test_augmented_rows():
    seq = np.zeros((100,17,3), dtype=np.float32)
    seq[:,0,0] = np.arange(100)
    x,y = make_augmented_rows([seq,seq], [1,0], ['duration','first','last'], fake_extractor)
    assert x.shape == (8,3) and y.tolist() == [1]*4+[0]*4
    assert np.allclose(x[:4], [[100,0,99],[90,10,99],[90,0,89],[80,10,89]])
    for bad in (np.zeros((5,17,3)),np.zeros((60,16,3))):
        try: make_augmented_rows([bad],[1],['duration','first','last'],fake_extractor)
        except ValueError: pass
        else: raise AssertionError('Bad sequence should be rejected')


if __name__=='__main__':
    test_augmented_rows()
    print('Crop10 full-data training tests: PASS (synthetic only)')
