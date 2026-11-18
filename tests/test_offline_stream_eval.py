"""eval_offline_stream.py on synthetic caches with a stand-in classifier."""
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

import eval_offline_stream as ev
from auto_detect_hailo_bridge import KP_DUMP_DTYPE
from auto_synthetic import missing, squat_rep, stand

FPS = 30.0


def write_cache(path, seq, video_id):
    rec = np.zeros(len(seq), dtype=KP_DUMP_DTYPE)
    rec['frame_id'] = np.arange(len(seq))
    rec['t'] = np.arange(len(seq)) / FPS
    rec['kp'] = np.stack(seq)
    rec['pose_valid'] = [0 if np.isnan(k[:, :2]).all() else 1 for k in seq]
    rec.tofile(path)
    Path(str(path) + '.json').write_text(json.dumps({
        'source': 'pc_ultralytics_cache', 'complete': True, 'assumed_fps': FPS, 'timebase': 'file',
        'video': {'video_id': video_id, 'sha256': 'x'}}))


def build(tmp_path, *, gap=0, reps=4, lead=30, tail=60, labels=None, video='PM_T', person=7):
    """Back-to-back toy squats; GT = motion frames of each rep."""
    seq, gt = [stand()] * lead, []
    for i in range(reps):
        start = len(seq)
        seq += squat_rep()[:45] + [stand()] * gap
        gt.append((i + 1, start, start + 44))
    seq += [stand()] * tail
    cache = tmp_path / 'cache'
    cache.mkdir(exist_ok=True)
    write_cache(cache / f'{video}.kp.bin', seq, video)
    seg = tmp_path / 'Segmentation.csv'
    new = not seg.exists()
    with seg.open('a', newline='') as f:
        w = csv.writer(f, delimiter=';')
        if new:
            w.writerow(['video_id', 'repetition_number', 'exercise_id', 'person_id',
                        'first_frame', 'last_frame', 'correctness'])
        for k, (rep, s, e) in enumerate(gt):
            w.writerow([video, rep, 6, person, s, e, (labels or [1, 0] * reps)[k]])
    return cache, seg, gt


def args_for(tmp_path, cache, seg, **kw):
    base = dict(cache_dir=cache, segmentation=seg, models_dir=tmp_path / 'models', in_sample_model=None,
                count_only=False, video=None, condition=None, out_dir=tmp_path / 'out', start_frame=0,
                end_frame=None, min_iou=0.2, mask_conf=0.3, board_mask_conf=0.0, bottom_depth=0.20,
                bottom_knee=120.0, pre_frames=16, post_frames=22, require_complete=True, run_name=None)
    base.update(kw)
    return Namespace(**base)


def always_good(seq, artifact):
    return 1, 0.9


def failing(seq, artifact):
    raise RuntimeError('No module named matplotlib')


# ------------------------------------------------------------ bug regressions
def test_unknown_posture_is_never_correct():
    gt = [{'rep': 1, 'start': 100, 'end': 140, 'label': 0, 'person_id': 1}]
    rep = {'detected_start_frame': 100, 'detected_end_frame': 140, 'start_frame': 84, 'end_frame': 162,
           'posture': 'unknown', 'classification_error': 'x'}
    rows, extras, _ = ev.evaluate_video('V', gt, [rep], lo=0, hi=10**6, min_iou=.2, condition='c', first_confirm=None)
    assert rows[0]['matched'] and rows[0]['correct'] is False and rows[0]['classified'] is False
    s = ev.summarize(rows, extras)
    assert s['failure_rate'] == 1.0 and s['accuracy_classified'] is None and s['accuracy_end_to_end'] == 0.0
    assert s['bad_recall_end_to_end'] == 0.0


def test_whole_cache_is_evaluated_by_default(tmp_path):
    cache, seg, gt = build(tmp_path, gap=60, reps=12)          # reps run past frame 1199
    assert gt[-1][2] > 1199
    out, summary = ev.run(args_for(tmp_path, cache, seg, condition=['no_overlap/off']),
                          classifier=always_good, models={'PM_T': {'held_out_person_id': 7}})
    s = summary['overall']['no_overlap/off']
    assert s['gt_reps'] == 12 and s['detected_matched'] == 12
    out2, summary2 = ev.run(args_for(tmp_path, cache, seg, condition=['no_overlap/off'], end_frame=1199),
                            classifier=always_good, models={'PM_T': {'held_out_person_id': 7}})
    assert summary2['overall']['no_overlap/off']['gt_reps'] < 12
    assert summary2['per_video']['no_overlap/off']['PM_T']['gt_excluded_by_range'] > 0


def test_matching_uses_detected_interval_not_classifier_window():
    gt = [{'rep': 1, 'start': 100, 'end': 140, 'label': 1, 'person_id': 1}]
    rep = {'detected_start_frame': 150, 'detected_end_frame': 190,      # detection elsewhere
           'start_frame': 95, 'end_frame': 212, 'posture': 'good'}      # window covers GT
    rows, extras, _ = ev.evaluate_video('V', gt, [rep], lo=0, hi=10**6, min_iou=.2, condition='c', first_confirm=None)
    assert rows[0]['matched'] is False and len(extras) == 1
    assert ev.iou((95, 212), (100, 140)) > 0.2                         # the old rule would have matched


# ------------------------------------------------------------ conditions
def test_overlap_policy_keeps_detection_and_restores_pre_context(tmp_path):
    cache, seg, gt = build(tmp_path, gap=0, reps=4)
    out, summary = ev.run(args_for(tmp_path, cache, seg), classifier=always_good,
                          models={'PM_T': {'held_out_person_id': 7}})
    rows = list(csv.DictReader((out / 'reps_eval.csv').open()))
    by = {}
    for r in rows:
        by.setdefault(r['condition'], []).append(r)
    det = {c: [(r['det_start'], r['det_end']) for r in rs] for c, rs in by.items()}
    assert det['no_overlap/off'] == det['overlap/off'] == det['no_overlap/on'] == det['overlap/on']
    s = summary['overall']
    assert s['no_overlap/off']['pre_context_negative'] >= 2 and s['overlap/off']['pre_context_negative'] == 0
    assert s['overlap/off']['pre_context_lt16'] == 0
    for a, b in zip(by['no_overlap/off'], by['overlap/off']):
        assert int(b['classification_frames']) >= int(a['classification_frames'])


def test_gating_on_recovers_reps_before_confirmation(tmp_path):
    cache, seg, gt = build(tmp_path, gap=15, reps=4)
    out, summary = ev.run(args_for(tmp_path, cache, seg, condition=['no_overlap/off', 'no_overlap/on']),
                          classifier=always_good, models={'PM_T': {'held_out_person_id': 7}})
    off, on = summary['overall']['no_overlap/off'], summary['overall']['no_overlap/on']
    assert on['detected_matched'] == off['detected_matched'] == 4
    assert on['matched_before_first_confirm'] >= 1                   # replayed first rep kept
    assert summary['per_video']['no_overlap/on']['PM_T']['squat_sessions'] == 1


def test_classification_failures_are_reported_separately(tmp_path):
    cache, seg, gt = build(tmp_path, gap=30, reps=3)
    out, summary = ev.run(args_for(tmp_path, cache, seg, condition=['overlap/off']),
                          classifier=failing, models={'PM_T': {'held_out_person_id': 7}})
    s = summary['overall']['overlap/off']
    assert s['detected_matched'] == 3 and s['classification_failed'] == 3 and s['failure_rate'] == 1.0
    assert s['accuracy_classified'] is None and s['accuracy_end_to_end'] == 0.0


def test_accuracy_definitions(tmp_path):
    cache, seg, gt = build(tmp_path, gap=30, reps=4, labels=[1, 0, 1, 0])
    out, summary = ev.run(args_for(tmp_path, cache, seg, condition=['overlap/off']),
                          classifier=always_good, models={'PM_T': {'held_out_person_id': 7}})
    s = summary['overall']['overlap/off']
    assert s['accuracy_classified'] == 0.5 and s['bad_recall_classified'] == 0.0
    assert s['confusion_classified'] == {'good_as_good': 2, 'good_as_bad': 0, 'bad_as_good': 2, 'bad_as_bad': 0}


def test_model_must_be_held_out_for_its_video(tmp_path):
    d = tmp_path / 'm'
    d.mkdir()
    joblib.dump({'held_out_video': 'PM_OTHER'}, d / 'PM_T.joblib')
    with pytest.raises(ValueError, match='held_out_video'):
        ev.load_models(d, ['PM_T'], None)


def test_exact_gap_counter():
    seq = [stand()] * 5 + [missing()] * 15 + [stand()] * 3 + [missing()] * 16 + [stand()]
    assert ev.exact_gap_runs(seq) == 1


def test_incomplete_cache_rejected(tmp_path):
    cache, seg, gt = build(tmp_path, reps=1)
    meta = cache / 'PM_T.kp.bin.json'
    m = json.loads(meta.read_text())
    m['complete'] = False
    meta.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='incomplete'):
        ev.run(args_for(tmp_path, cache, seg), classifier=always_good,
               models={'PM_T': {'held_out_person_id': 7}})


def test_relative_out_dir_stays_under_cwd(tmp_path, monkeypatch):
    cache, seg, gt = build(tmp_path, reps=2, gap=30)
    monkeypatch.chdir(tmp_path)
    out, _ = ev.run(args_for(tmp_path, cache, seg, out_dir=Path('rel_out'), run_name='r',
                             condition=['no_overlap/on']),
                    classifier=always_good, models={'PM_T': {'held_out_person_id': 7}})
    assert out == (tmp_path / 'rel_out' / 'r').resolve()
    assert (out / 'no_overlap__gating_on' / 'PM_T_auto_events.jsonl').is_file()
