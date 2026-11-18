"""train_lopo_heldout_models.py on a tiny synthetic corpus (no REHAB24-6 data)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

import train_lopo_heldout_models as tl
from auto_synthetic import stand
from train_squat_crop10_all9 import make_augmented_rows
from train_squat_extended import extract_feature_groups

TEMPLATE = ROOT / 'models' / 'squat_plus_asymmetry180.joblib'


def rep_sequence(depth, lean):
    frames = []
    for d in list(np.linspace(0, depth, 20)) + [depth] * 5 + list(np.linspace(depth, 0, 20)):
        kp = stand(400 + 200 * d).copy()
        kp[[0, 5, 6], 0] += lean * d * 100          # torso lean grows with depth
        kp[:, 2] = 0.95
        kp[[1, 2, 3, 4, 7, 8], :] = [0, 0, 0]       # unused joints: zero confidence
        frames.append(kp)
    return np.stack(frames).astype(np.float32)


@pytest.fixture
def corpus(tmp_path):
    rows = []
    rng = np.random.default_rng(0)
    for p, video in enumerate(['PM_A', 'PM_B', 'PM_C'], start=1):
        for r in range(1, 7):
            label = r % 2
            seq = rep_sequence(0.6 if label else 0.3, 0.1 if label else 0.6) + rng.normal(0, .5, (45, 17, 3)).astype(np.float32) * [1, 1, 0]
            f = f'{video}_rep{r:02d}.npz'
            np.savez(tmp_path / f, keypoints=seq, label=label, person_id=p)
            rows.append({'file': f, 'person_id': p, 'video_id': video, 'label': label, 'repetition_number': r})
    pd.DataFrame(rows).to_csv(tmp_path / 'index.csv', index=False)
    return tmp_path


def test_folds_exclude_held_out_person(corpus, tmp_path):
    index, seqs = tl.load_corpus(corpus, expect_reps=18, expect_people=3)
    template = joblib.load(TEMPLATE)
    names = template['feature_names']
    out = tmp_path / 'models'
    for variant, crop in tl.VARIANTS.items():
        preds = tl.fit_variant(variant, crop, index, seqs, names, template['pipeline'], out,
                               extract_feature_groups, {'note': 'test'})
        assert len(preds) == 18
        for video, person in [('PM_A', 1), ('PM_B', 2), ('PM_C', 3)]:
            art = joblib.load(out / variant / f'{video}.joblib')
            assert art['held_out_person_id'] == person and person not in art['training_person_ids']
            assert art['training_feature_rows'] == (48 if crop else 12)   # 12 reps x4 crops
        manifest = json.loads((out / variant / 'manifest.json').read_text())
        assert len(manifest['folds']) == 3


def test_verify_all9_matches_the_deployed_recipe(corpus, tmp_path):
    index, seqs = tl.load_corpus(corpus, expect_reps=18, expect_people=3)
    template = joblib.load(TEMPLATE)
    names = template['feature_names']
    x, y = make_augmented_rows(seqs, index.label.astype(int).tolist(), names)
    deployed = {'pipeline': clone(template['pipeline']).fit(x, y)}
    joblib.dump(deployed, tmp_path / 'deployed.joblib')
    result = tl.verify_all9(index, seqs, names, template['pipeline'], tmp_path / 'deployed.joblib',
                            extract_feature_groups)
    assert result['pass'] and result['max_abs_coef_diff'] == 0.0
    # a model fitted differently must fail the check
    other = {'pipeline': clone(template['pipeline']).fit(x[::4], y[::4])}
    joblib.dump(other, tmp_path / 'other.joblib')
    assert tl.verify_all9(index, seqs, names, template['pipeline'], tmp_path / 'other.joblib',
                          extract_feature_groups)['max_abs_coef_diff'] > 0


def test_refuses_to_overwrite(corpus, tmp_path, monkeypatch):
    (tmp_path / 'models' / 'PlusAsymmetry180').mkdir(parents=True)
    monkeypatch.setattr(sys, 'argv', ['x', '--data-dir', str(corpus), '--out-dir', str(tmp_path / 'models'),
                                      '--template-model', str(TEMPLATE), '--expect-reps', '18',
                                      '--expect-people', '3'])
    with pytest.raises(FileExistsError):
        tl.main()
