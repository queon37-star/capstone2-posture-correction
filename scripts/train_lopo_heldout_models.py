"""Re-create person-held-out (LOPO) 180-feature Logistic Regression models.

For each of the nine REHAB24-6 Ex6 Camera17 videos (= nine people), fit the
existing PlusAsymmetry180 pipeline on the OTHER eight people's manually
segmented repetitions and save it for offline stream evaluation.

Variants (same recipe as the existing scripts, nothing re-tuned):
  PlusAsymmetry180         no augmentation   (as train_squat_lopo_stream_models.py)
  PlusAsymmetry180_crop10  Crop10 x4 rows    (as train_squat_crop10_all9.py, the deployed recipe)

Also writes a GT-segment LOPO reference evaluation (full manual segments; this
is NOT the online-buffer condition) and, with --verify-all9, re-fits Crop10 on
all nine people and compares it with models/squat_plus_asymmetry180_crop10_all9.joblib
to confirm the recipe reproduces the deployed model.

Reads data/processed + models/*.joblib only; writes only under --out-dir.

  python scripts/train_lopo_heldout_models.py --verify-all9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_squat_lopo_augmentation import extract_vector  # noqa: E402
from train_squat_crop10_all9 import make_augmented_rows  # noqa: E402
from train_squat_lopo_stream_models import validate_metadata  # noqa: E402

VARIANTS = {'PlusAsymmetry180': 0.0, 'PlusAsymmetry180_crop10': 0.10}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_corpus(data_dir, expect_reps=195, expect_people=9):
    index = pd.read_csv(data_dir / 'index.csv')
    validate_metadata(index)
    if expect_reps and len(index) != expect_reps:
        raise ValueError(f'expected {expect_reps} repetitions, found {len(index)}')
    if expect_people and index.person_id.nunique() != expect_people:
        raise ValueError(f'expected {expect_people} people, found {index.person_id.nunique()}')
    sequences = []
    for row in index.itertuples(index=False):
        path = data_dir / str(row.file)
        with np.load(path, allow_pickle=False) as pack:
            seq = np.asarray(pack['keypoints'], dtype=np.float32)
            if int(np.asarray(pack['label']).item()) != int(row.label) or \
               int(np.asarray(pack['person_id']).item()) != int(row.person_id):
                raise ValueError(f'index/NPZ mismatch: {path}')
        sequences.append(seq)
    return index, sequences


def training_rows(sequences, labels, names, crop, extractor):
    if crop > 0:
        return make_augmented_rows(sequences, labels, names, extractor=extractor, fraction=crop)
    x = np.stack([extract_vector(s, names, extractor) for s in sequences]).astype(np.float64)
    return x, np.asarray(labels, dtype=int)


def binary_report(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp_bad = int(((y == 0) & (pred == 0)).sum())
    fn_bad = int(((y == 0) & (pred == 1)).sum())
    fp_bad = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 1) & (pred == 1)).sum())
    rec_bad = tp_bad / max(tp_bad + fn_bad, 1)
    rec_good = tn / max(tn + fp_bad, 1)
    return {'n': int(len(y)), 'correct': tp_bad + tn, 'accuracy': round((tp_bad + tn) / max(len(y), 1), 4),
            'bad_recall': round(rec_bad, 4), 'bad_precision': round(tp_bad / max(tp_bad + fp_bad, 1), 4),
            'balanced_accuracy': round((rec_bad + rec_good) / 2, 4),
            'confusion': {'good_as_good': tn, 'good_as_bad': fp_bad, 'bad_as_good': fn_bad, 'bad_as_bad': tp_bad}}


def fit_variant(variant, crop, index, sequences, names, template, out_dir, extractor, provenance):
    out = out_dir / variant
    out.mkdir(parents=True, exist_ok=True)
    labels = index.label.astype(int).to_numpy()
    preds, manifest = [], []
    for video in sorted(index.video_id.unique()):
        person = int(index.loc[index.video_id == video, 'person_id'].iloc[0])
        train = (index.person_id != person).to_numpy()
        tr_idx = np.flatnonzero(train)
        x, y = training_rows([sequences[i] for i in tr_idx], labels[tr_idx], names, crop, extractor)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(f'{variant}/{video}: training lacks a class')
        model = clone(template).fit(x, y)
        train_people = sorted(int(p) for p in index.loc[train, 'person_id'].unique())
        assert person not in train_people
        artifact = {
            'pipeline': model, 'feature_names': list(names), 'feature_groups': ['base', 'asymmetry'],
            'feature_variant': 'PlusAsymmetry180', 'training_variant': f'{variant}_lopo',
            'crop_fraction': crop, 'label_map': {0: 'incorrect', 1: 'correct'},
            'held_out_video': str(video), 'held_out_person_id': person,
            'training_person_ids': train_people, 'training_repetitions': int(train.sum()),
            'training_feature_rows': int(len(y)),
            'evaluation_note': 'Held-out weights only; features/thresholds/augmentation were chosen on all nine people.',
            **provenance}
        path = out / f'{video}.joblib'
        joblib.dump(artifact, path)
        test_idx = np.flatnonzero(~train)
        x_test = np.stack([extract_vector(sequences[i], names, extractor) for i in test_idx])
        p = model.predict(x_test).astype(int)
        prob = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        for i, pi, pr in zip(test_idx, p, prob):
            r = index.iloc[i]
            preds.append({'variant': variant, 'video_id': r.video_id, 'person_id': int(r.person_id),
                          'repetition_number': int(r.repetition_number), 'label': int(r.label),
                          'pred': int(pi), 'prob_correct': round(float(pr), 5)})
        manifest.append({'held_out_video': str(video), 'held_out_person_id': person,
                         'training_repetitions': int(train.sum()), 'training_feature_rows': int(len(y)),
                         'model_path': str(path)})
        print(f'{variant}: {video} (person {person}) train={train.sum()} reps/{len(y)} rows -> {path}', flush=True)
    (out / 'manifest.json').write_text(json.dumps({'variant': variant, 'crop_fraction': crop,
                                                   'folds': manifest, **provenance},
                                                  ensure_ascii=False, indent=1), encoding='utf-8')
    return preds


def verify_all9(index, sequences, names, template, deployed_path, extractor):
    labels = index.label.astype(int).to_numpy()
    x, y = make_augmented_rows(sequences, labels, names, extractor=extractor, fraction=0.10)
    refit = clone(template).fit(x, y)
    deployed = joblib.load(deployed_path)['pipeline']
    x_full = np.stack([extract_vector(s, names, extractor) for s in sequences])
    p_refit = refit.predict_proba(x_full)[:, 1]
    p_dep = deployed.predict_proba(x_full)[:, 1]
    coef_diff = float(np.max(np.abs(refit.named_steps['model'].coef_
                                    - deployed.named_steps['model'].coef_)))
    result = {'deployed_model': str(deployed_path), 'rows': int(len(y)),
              'max_abs_prob_diff': float(np.max(np.abs(p_refit - p_dep))),
              'max_abs_coef_diff': coef_diff,
              'same_predictions': bool(np.array_equal(refit.predict(x_full), deployed.predict(x_full)))}
    result['pass'] = result['same_predictions'] and result['max_abs_prob_diff'] < 1e-6
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path, default=Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'))
    ap.add_argument('--template-model', type=Path, default=Path('models/squat_plus_asymmetry180.joblib'))
    ap.add_argument('--deployed-model', type=Path, default=Path('models/squat_plus_asymmetry180_crop10_all9.joblib'))
    ap.add_argument('--out-dir', type=Path, default=Path('models/lopo_heldout'))
    ap.add_argument('--variant', action='append', choices=sorted(VARIANTS))
    ap.add_argument('--verify-all9', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--expect-reps', type=int, default=195)
    ap.add_argument('--expect-people', type=int, default=9)
    args = ap.parse_args()

    variants = args.variant or list(VARIANTS)
    for v in variants:
        if (args.out_dir / v).exists() and not args.overwrite:
            raise FileExistsError(f'{args.out_dir / v} exists (use --overwrite or another --out-dir)')
    from train_squat_extended import extract_feature_groups
    source = joblib.load(args.template_model)  # self-trained artifact only
    names = list(source['feature_names'])
    if source.get('feature_variant') != 'PlusAsymmetry180' or len(set(names)) != 180:
        raise ValueError('template must be the PlusAsymmetry180 180-feature model')
    index, sequences = load_corpus(args.data_dir, args.expect_reps, args.expect_people)
    provenance = {'data_index_sha256': sha256_file(args.data_dir / 'index.csv'),
                  'template_model': str(args.template_model),
                  'template_sha256': sha256_file(args.template_model),
                  'sklearn_version': sklearn.__version__,
                  'trained_at_utc': datetime.now(timezone.utc).isoformat()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_preds, summary = [], {'note': 'GT-segment LOPO reference (full manual segments, not online buffer windows)'}
    for v in variants:
        preds = fit_variant(v, VARIANTS[v], index, sequences, names, source['pipeline'],
                            args.out_dir, extract_feature_groups, provenance)
        all_preds += preds
        summary[v] = binary_report([p['label'] for p in preds], [p['pred'] for p in preds])
        print(v, json.dumps(summary[v]), flush=True)
    if args.verify_all9:
        summary['verify_all9'] = verify_all9(index, sequences, names, source['pipeline'],
                                             args.deployed_model, extract_feature_groups)
        print('verify_all9', json.dumps(summary['verify_all9']), flush=True)
    pd.DataFrame(all_preds).to_csv(args.out_dir / 'gt_segment_lopo_predictions.csv', index=False)
    (args.out_dir / 'gt_segment_lopo_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    enforce_verification(summary, enabled=args.verify_all9)


def enforce_verification(summary, *, enabled):
    # Results are written first, so mismatches remain inspectable on disk.
    if enabled and not summary.get('verify_all9', {}).get('pass', False):
        raise SystemExit('FAIL: --verify-all9 did not reproduce the deployed model. '
                         'Inspect gt_segment_lopo_summary.json before using these results.')


if __name__ == '__main__':
    main()
