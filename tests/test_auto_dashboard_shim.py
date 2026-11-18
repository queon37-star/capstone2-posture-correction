"""auto_dashboard_shim: factory wiring, classifier preload switch and /status payload."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

import auto_dashboard_shim as shim
import squat_stream_pipeline
from auto_synthetic import FPS, squats, stand


@pytest.fixture
def hooked(monkeypatch, tmp_path):
    monkeypatch.setenv('SQUAT_AUTO_TIMEBASE', 'file')
    monkeypatch.setenv('SQUAT_AUTO_OUTPUT', str(tmp_path / 'events.jsonl'))
    monkeypatch.setenv('SQUAT_COUNT_ONLY', '1')
    # monkeypatch restores the original class after the test
    monkeypatch.setattr(squat_stream_pipeline, 'SquatStreamPipeline',
                        squat_stream_pipeline.SquatStreamPipeline)
    monkeypatch.setitem(shim._STATE, 'pipeline', None)

    def build(gating):
        monkeypatch.setenv('SQUAT_AUTO_GATING', str(gating))
        shim.install_pipeline_hook()
        return squat_stream_pipeline.SquatStreamPipeline(
            artifact=None, fps=FPS, bottom_depth=0.20, count_only=True)
    return build


@pytest.mark.parametrize('gating', [0, 1])
def test_factory_and_status(hooked, gating):
    p = hooked(gating)
    assert p.gating is bool(gating) and p.make_base is not None
    assert shim._STATE['classifier_preload'] == 'skipped_count_only'
    for f, kp in enumerate([stand()] * 30 + squats(2)):
        p.push(kp, frame_id=f)
    status = shim._status()
    json.dumps(status, default=str)            # must stay serialisable
    assert status['gating'] is bool(gating)
    assert status['squat_count'] == p.counter.count
    assert status['published_reps'] == p.event_count
    if gating:
        assert {'session_id', 'session_count', 'total_count', 'last_replay'} <= set(status)
        assert p.base is not p.make_base()     # make_base builds a new object each time


def test_preload_in_classifier_mode(monkeypatch):
    monkeypatch.setenv('SQUAT_COUNT_ONLY', '0')
    monkeypatch.setitem(shim._STATE, 'classifier_preload', 'not_requested')
    shim.preload_classifier()
    assert shim._STATE['classifier_preload'].startswith(('ok', 'error'))
