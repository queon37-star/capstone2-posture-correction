"""Review regressions: multi-session replay accounting and all9 failure reporting."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import eval_offline_stream as ev
from train_lopo_heldout_models import enforce_verification


def test_recovery_attributed_to_each_session_not_first_only():
    gt = [
        {'rep': 1, 'start': 20, 'end': 40, 'label': 1, 'person_id': 7},
        {'rep': 2, 'start': 220, 'end': 240, 'label': 1, 'person_id': 7},
    ]
    reps = [
        {'detected_start_frame': 20, 'detected_end_frame': 40,
         'start_frame': 10, 'end_frame': 62, 'posture': 'good', 'session_id': 1},
        {'detected_start_frame': 220, 'detected_end_frame': 240,
         'start_frame': 210, 'end_frame': 262, 'posture': 'good', 'session_id': 2},
    ]
    rows, extras, _ = ev.evaluate_video('V', gt, reps, lo=0, hi=500,
        min_iou=.2, condition='no_overlap/on', session_confirms={1: 45, 2: 245})
    assert [r['before_session_confirm'] for r in rows] == [True, True]
    stats = ev.summarize(rows, extras)
    assert stats['matched_before_session_confirm'] == 2
    assert stats['matched_with_known_session_confirm'] == 2
    assert stats['matched_before_first_confirm'] == 2


def test_ungated_reps_are_not_counted_as_replayed():
    gt = [{'rep': 1, 'start': 20, 'end': 40, 'label': 1, 'person_id': 7}]
    reps = [{'detected_start_frame': 20, 'detected_end_frame': 40,
             'start_frame': 10, 'end_frame': 62, 'posture': 'good'}]
    rows, extras, _ = ev.evaluate_video('V', gt, reps, lo=0, hi=500,
        min_iou=.2, condition='no_overlap/off', session_confirms=None)
    assert rows[0]['before_session_confirm'] is None
    assert ev.summarize(rows, extras)['matched_before_session_confirm'] == 0


def test_verify_all9_mismatch_is_nonzero_exit():
    enforce_verification({'verify_all9': {'pass': True}}, enabled=True)
    enforce_verification({'verify_all9': {'pass': False}}, enabled=False)
    with pytest.raises(SystemExit, match='FAIL: --verify-all9'):
        enforce_verification({'verify_all9': {'pass': False}}, enabled=True)
