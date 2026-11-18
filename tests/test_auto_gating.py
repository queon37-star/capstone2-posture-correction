"""SQUAT_AUTO_GATING tests with synthetic keypoints (no Hailo, camera or model).

Most tests use ScriptedDetector so a failure points at the gating logic, not at
ExerciseDetector thresholds. Synthetic success is NOT evidence of real-world
exercise-recognition accuracy.
"""
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

from auto_detect_hailo_bridge import AutoDetectPipeline
from auto_synthetic import FPS, ScriptedDetector, missing, plank, pushups, squats, stand
from squat_stream_pipeline import SquatStreamPipeline

FRAME_KEYS = ('start_frame', 'end_frame', 'detected_start_frame', 'detected_end_frame')


def make_base():
    return SquatStreamPipeline(artifact=None, fps=FPS, bottom_depth=0.20, count_only=True)


@pytest.fixture(autouse=True)
def _file_timebase(monkeypatch):
    monkeypatch.setenv('SQUAT_AUTO_TIMEBASE', 'file')
    monkeypatch.delenv('SQUAT_AUTO_GATING', raising=False)


def run(tmp_path, seq, *, gating, changes=None, name='run', **kwargs):
    detector = ScriptedDetector(changes) if changes is not None else None
    p = AutoDetectPipeline(make_base(), FPS, events_path=tmp_path / f'{name}.jsonl',
                           make_base=make_base, gating=gating, detector=detector, **kwargs)
    reps = []
    for f, kp in enumerate(seq):
        reps += p.push(kp, frame_id=f)
    p.before_flush = {'stand_hip': p.counter.stand_hip, 'state': p.counter.state}
    reps += p.flush()
    events = [json.loads(line) for line in (tmp_path / f'{name}.jsonl').read_text().splitlines()]
    return p, reps, events


def bare(seq):
    pipe = make_base()
    reps = []
    for f, kp in enumerate(seq):
        reps += pipe.push(kp, frame_id=f)
    return reps + pipe.flush()


def spans(reps):
    return [(r['detected_start_frame'], r['detected_end_frame']) for r in reps]


# ---------------------------------------------------------------- gating OFF
def test_gating_is_off_by_default(tmp_path):
    p = AutoDetectPipeline(make_base(), FPS, events_path=tmp_path / 'e.jsonl')
    assert p.gating is False
    assert p.counter is p.base.counter and p.event_count == p.base.event_count


def test_gating_off_matches_bare_pipeline(tmp_path):
    seq = [stand()] * 30 + squats(3) + [missing()] * 30 + [plank()] * 30 + pushups(4) + [stand()] * 40 + squats(2)
    _, reps, _ = run(tmp_path, seq, gating=0)
    assert reps == bare(seq)            # identical dicts, including 'count'


def test_gating_requires_factory(tmp_path):
    with pytest.raises(ValueError):
        AutoDetectPipeline(make_base(), FPS, events_path=tmp_path / 'e.jsonl', gating=1)


# ---------------------------------------------------------------- gating ON
def test_1_first_rep_before_confirmation_is_replayed(tmp_path):
    seq = [stand()] * 30 + squats(3)
    first_end = bare(seq)[0]['detected_end_frame']
    confirm = first_end + 10
    p, reps, _ = run(tmp_path, seq, gating=1, changes=[(0, 'idle'), (confirm, 'squat')])
    assert len(reps) == 3
    assert reps[0]['detected_end_frame'] < confirm
    for key in FRAME_KEYS:                     # same boundaries as the ungated stream
        assert [r[key] for r in reps] == [r[key] for r in bare(seq)]
    assert all(r['pre_context_complete'] for r in reps)
    assert p.last_replay['first_frame'] == 0


def test_2_no_squat_reps_during_pushup_session(tmp_path):
    seq = [plank()] * 30 + pushups(6)
    _, ungated, _ = run(tmp_path, seq, gating=0, changes=[(0, 'idle'), (60, 'pushup')], name='a')
    p, gated, events = run(tmp_path, seq, gating=1, changes=[(0, 'idle'), (60, 'pushup')], name='b')
    assert len(ungated) >= 1          # the bug being fixed: plank-calibrated counter counts push-ups
    assert gated == [] and p.counter.count == 0
    assert [e['type'] for e in events if e['type'] != 'label_change'] == ['session_start', 'session_end']


def test_3_fast_pushup_to_squat_excludes_plank_frames(tmp_path):
    seq = [plank()] * 30 + pushups(3) + [stand()] * 30 + squats(3)
    last_plank = 30 + 165 - 1
    confirm = 300                              # replay window (6 s) reaches back into the plank
    p, reps, _ = run(tmp_path, seq, gating=1,
                     changes=[(0, 'idle'), (40, 'pushup'), (confirm, 'squat')])
    assert confirm - p.replay_sec * FPS < last_plank
    assert p.last_replay['first_frame'] > last_plank
    assert len(reps) == 3
    assert all(s > last_plank for s, _ in spans(reps))
    assert p.before_flush['stand_hip'] == pytest.approx(400, abs=2)   # calibrated standing, not on the floor


def test_3b_plank_calibrated_counter_misses_squats_without_gating(tmp_path):
    """Documents why a fresh counter is needed (an output filter alone is not enough)."""
    seq = [plank()] * 30 + pushups(3) + [stand()] * 30 + squats(3)
    reps = bare(seq)
    squat_reps = [r for r in reps if r['detected_start_frame'] > 30 + 165]
    assert len(squat_reps) < 3


def test_4_squat_to_pushup_flushes_pending_and_drops_unfinished(tmp_path):
    down = [stand(400 + 200 * d) for d in [i * 0.6 / 19 for i in range(20)]]
    rep = down + [stand(520)] * 5 + down[::-1]
    seq = [stand()] * 30 + rep + [stand()] * 15 + rep + [stand()] * 6 + rep + [stand()] * 15
    ungated = bare(seq)
    assert len(ungated) == 3                    # the third descent is a real rep if allowed to finish
    switch = ungated[1]['detected_end_frame'] + 15   # rep 2 still lacks 22 post frames,
    assert ungated[2]['detected_start_frame'] < switch < ungated[2]['detected_end_frame']  # rep 3 in progress
    p, reps, events = run(tmp_path, seq, gating=1,
                          changes=[(0, 'idle'), (40, 'squat'), (switch, 'pushup')])
    assert spans(reps) == spans(ungated[:2])
    assert reps[0]['post_context_complete'] is True
    assert reps[1]['post_context_complete'] is False      # flushed at session end
    end = [e for e in events if e['type'] == 'session_end'][0]
    assert end['exercise'] == 'squat' and end['session_reps'] == 2


def test_5a_short_dropout_inside_squat_session(tmp_path):
    seq = [stand()] * 30 + squats(2) + [missing()] * 8 + [stand()] * 10 + squats(2)
    p, reps, events = run(tmp_path, seq, gating=1, changes=[(0, 'idle'), (40, 'squat')])
    assert len(reps) == 4 == len(bare(seq))
    assert [e['type'] for e in events].count('session_start') == 1


def test_5b_person_lost_then_reappears_recalibrates(tmp_path):
    part1 = [stand()] * 30 + squats(2)                     # frames 0..149
    gap = [missing()] * 120                                # 4 s > SQUAT_AUTO_LOST_SEC
    part2 = [stand(450, x0=800)] * 30 + squats_at(450, 800, 2)
    seq = part1 + gap + part2
    gap_end = len(part1) + len(gap) - 1
    p, reps, events = run(tmp_path, seq, gating=1, changes=[
        (0, 'idle'), (40, 'squat'), (gap_end + 75, 'squat')])
    ends = [e for e in events if e['type'] == 'session_end']
    assert ends[0]['reason'] == 'person lost'
    assert p.last_replay['first_frame'] > gap_end
    assert len(reps) == 4 and [r['session_id'] for r in reps] == [1, 1, 2, 2]
    assert p.before_flush['stand_hip'] == pytest.approx(450, abs=2)


def test_5c_pushup_person_lost_then_squat(tmp_path):
    seq = [plank()] * 30 + pushups(3) + [missing()] * 120 + [stand()] * 30 + squats(2)
    gap_end = 30 + 165 + 120 - 1
    p, reps, _ = run(tmp_path, seq, gating=1, changes=[
        (0, 'idle'), (40, 'pushup'), (gap_end + 75, 'squat')])
    assert len(reps) == 2 and all(s > gap_end for s, _ in spans(reps))
    assert p.last_replay['first_frame'] > gap_end


def test_6_session_and_total_counts(tmp_path):
    seq = [stand()] * 30 + squats(3) + [stand()] * 150 + squats(2) + [stand()] * 60
    p, reps, events = run(tmp_path, seq, gating=1, changes=[
        (0, 'idle'), (40, 'squat'), (260, 'idle'), (420, 'squat'), (560, 'idle')])
    assert [r['count'] for r in reps] == list(range(1, len(reps) + 1))
    assert [(r['session_id'], r['session_count']) for r in reps] == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    assert p.event_count == p.total_count == 5 and p.counter.count == 5
    assert p.counter.state == 'INACTIVE' and p.session_last_rep['session_id'] == 2
    ends = [e for e in events if e['type'] == 'session_end']
    assert [e['session_reps'] for e in ends] == [3, 2]


def test_squat_idle_squat_does_not_double_count(tmp_path):
    seq = [stand()] * 30 + squats(3) + [stand()] * 45 + squats(2)
    p, reps, _ = run(tmp_path, seq, gating=1, changes=[
        (0, 'idle'), (100, 'squat'), (220, 'idle'), (330, 'squat')])
    assert len(reps) == 5 and len(set(spans(reps))) == 5
    assert spans(reps) == spans(bare(seq))


def test_replay_from_mid_stream_clamps_pre_context_without_error(tmp_path):
    """Cut right before the first rep: fewer than 16 pre-context frames exist.
    Relative frame ids make RepBoundaryBuffer clamp as at stream start."""
    seq = [missing()] * 30 + [stand()] * 12 + squats(2)
    p, reps, _ = run(tmp_path, seq, gating=1, changes=[(0, 'idle'), (130, 'squat')])
    assert len(reps) == 2
    assert reps[0]['pre_context_complete'] is False and reps[0]['start_frame'] == 30
    assert reps[1]['pre_context_complete'] is True
    # The naive alternative (fresh pipeline fed absolute frame ids) fails:
    naive = make_base()
    with pytest.raises(RuntimeError, match='History does not cover'):
        for f, kp in enumerate(seq[30:], start=30):
            naive.push(kp, frame_id=f)


def test_real_detector_smoke(tmp_path):
    """ExerciseDetector on toy data: push-ups counted only without gating."""
    seq = ([stand()] * 40 + squats(4) + [stand()] * 60 + [missing()] * 30 + [plank()] * 30
           + pushups(6) + [missing()] * 120 + [stand()] * 60 + squats(3) + [stand()] * 150)
    _, ungated, _ = run(tmp_path, seq, gating=0, name='a')
    _, gated, _ = run(tmp_path, seq, gating=1, name='b')
    assert len(ungated) == 13 and len(gated) == 7
    assert set(spans(gated)) <= set(spans(ungated))


def test_no_back_to_back_label_change_events(tmp_path):
    """Guard for the 340-frame idle burst seen in c922_combo_events.jsonl."""
    seq = ([stand()] * 120 + ([stand()] * 20 + [missing()] * 10) * 20 + [missing()] * 200
           + [stand()] * 200)
    for gating in (0, 1):
        _, _, events = run(tmp_path, seq, gating=gating, name=f'g{gating}')
        frames = [e['frame_id'] for e in events if e['type'] == 'label_change']
        assert all(b - a > 1 for a, b in zip(frames, frames[1:]))


def test_7_replay_cost_is_reported(tmp_path):
    seq = [stand()] * 30 + squats(5)
    p, _, _ = run(tmp_path, seq, gating=1, changes=[(0, 'idle'), (len(seq) - 1, 'squat')])
    assert p.last_replay['frames'] == 180
    assert 0 < p.last_replay['ms'] < 1000      # generous; real numbers come from the board log


def squats_at(hip, x0, n):
    out = []
    for _ in range(n):
        down = [stand(hip + 200 * d, x0) for d in [i * 0.6 / 19 for i in range(20)]]
        out += down + [stand(hip + 120, x0)] * 5 + down[::-1] + [stand(hip, x0)] * 15
    return out
