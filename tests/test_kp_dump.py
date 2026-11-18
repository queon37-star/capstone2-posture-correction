"""SQUAT_KP_DUMP round trip and 'no behaviour change' checks (no Hailo needed)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

from auto_detect_hailo_bridge import AutoDetectPipeline, KP_DUMP_DTYPE, read_kp_dump
from auto_synthetic import FPS, missing, squats, stand
from squat_stream_pipeline import SquatStreamPipeline


def make_base():
    return SquatStreamPipeline(artifact=None, fps=FPS, bottom_depth=0.20, count_only=True)


def sequence():
    return [stand()] * 30 + squats(3) + [missing()] * 20 + [stand()] * 30 + squats(2)


def run_wrapper(tmp_path, seq, **kwargs):
    p = AutoDetectPipeline(make_base(), FPS, events_path=tmp_path / 'events.jsonl', **kwargs)
    reps = []
    for f, kp in enumerate(seq):
        reps += p.push(kp, frame_id=f)
    reps += p.flush()
    return p, reps


def test_record_layout_is_fixed():
    assert KP_DUMP_DTYPE.itemsize == 228


def test_dump_round_trip_and_missing_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('SQUAT_AUTO_TIMEBASE', 'file')
    seq = sequence()
    _, _ = run_wrapper(tmp_path, seq, kp_dump_path=tmp_path / 'kp.bin')
    data = read_kp_dump(tmp_path / 'kp.bin')
    assert len(data) == len(seq)
    assert np.array_equal(data['frame_id'], np.arange(len(seq)))
    assert np.allclose(data['t'], np.arange(len(seq)) / FPS)
    stacked = np.stack(seq)
    assert np.array_equal(np.isnan(data['kp']), np.isnan(stacked))
    assert np.allclose(np.nan_to_num(data['kp']), np.nan_to_num(stacked))
    missing_rows = [i for i, kp in enumerate(seq) if np.isnan(kp[:, :2]).all()]
    assert set(np.flatnonzero(data['pose_valid'] == 0)) == set(missing_rows)
    meta = json.loads((tmp_path / 'kp.bin.json').read_text(encoding='utf-8'))
    assert meta['frames'] == len(seq) and meta['record_bytes'] == 228


def test_partial_file_is_readable(tmp_path, monkeypatch):
    """A Ctrl+C mid-write leaves a truncated last record; the reader ignores it."""
    monkeypatch.setenv('SQUAT_AUTO_TIMEBASE', 'file')
    run_wrapper(tmp_path, sequence()[:100], kp_dump_path=tmp_path / 'kp.bin')
    raw = (tmp_path / 'kp.bin').read_bytes()
    (tmp_path / 'cut.bin').write_bytes(raw[:-50])
    assert len(read_kp_dump(tmp_path / 'cut.bin')) == 99


def test_dump_does_not_change_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv('SQUAT_AUTO_TIMEBASE', 'file')
    seq = sequence()
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    _, reps_off = run_wrapper(tmp_path / 'a', seq)
    _, reps_on = run_wrapper(tmp_path / 'b', seq, kp_dump_path=tmp_path / 'b' / 'kp.bin')
    assert reps_off == reps_on and len(reps_on) == 5
    assert ((tmp_path / 'a' / 'events.jsonl').read_text()
            == (tmp_path / 'b' / 'events.jsonl').read_text())
