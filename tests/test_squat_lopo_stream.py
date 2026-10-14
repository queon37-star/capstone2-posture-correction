"""Synthetic leakage check for fold construction, no private dataset needed."""
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_squat_lopo_stream_models import fit_heldout_models, validate_metadata
from summarize_squat_stream_evals import aggregate


def test_training_excludes_person():
    # Three synthetic participants, two labels each. Training is refitted per fold.
    meta = pd.DataFrame([
        {'file': f'vid{p}_{r}.npz', 'person_id':p, 'video_id': f'PM_{p:03d}',
         'repetition_number':r, 'label':int(r%2)}
        for p in (1, 2, 3) for r in range(8)
    ])
    validate_metadata(meta)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(len(meta), 4))
    x[:, 0] += 2*meta.label.to_numpy()
    template = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                             LogisticRegression(max_iter=300, random_state=0))
    # Pretend user supplied full-data fitted artifact; clone must discard weights.
    template.fit(x, meta.label)
    with tempfile.TemporaryDirectory() as d:
        folds = fit_heldout_models(meta, x, ['f1','f2','f3','f4'], template, Path(d))
        assert len(folds) == 3
        for f in folds:
            saved = joblib.load(f['model_path'])
            assert saved['held_out_person_id'] not in saved['training_person_ids']
            assert saved['training_sample_count'] == 16
            assert saved['held_out_sample_count'] == 8
            assert not np.shares_memory(saved['pipeline'][-1].coef_, template[-1].coef_)


def test_aggregate():
    one = {
      'gt_repetitions':3, 'predicted_repetitions':3, 'matched':3,
      'missed':0,'extra':0,'gt_incorrect':1,'gt_correct':2,
      'detected_and_classified_incorrect':1,'detected_and_classified_correct':1,
      'matched_pred_bad':2,'extra_pred_bad':0,'avg_interval_iou':.8,
    }
    result = aggregate([one, one])
    assert result['matched_correct'] == 4
    assert result['matched_classifier_accuracy'] == 4/6
    assert result['incorrect_end_to_end_recall'] == 1.0
    assert result['bad_precision_including_extra_events'] == .5


if __name__ == '__main__':
    test_training_excludes_person()
    test_aggregate()
    print('LOPO classifier fold/isolation and aggregator tests: PASS (synthetic)')
