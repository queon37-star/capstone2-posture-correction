"""SquatCounter must recover after a hip/pose gap of any length (14/15/16/20 frames).

Before the fix, a gap of exactly max(10, fps*0.5) = 15 frames left the counter in
READY with scale=NaN, so it never counted again.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

from auto_synthetic import missing, squats, stand
from squat_stream_pipeline import SquatStreamPipeline
from stream_squat_core import SquatCounter


def count(seq, bottom_depth=0.20):
    c = SquatCounter(fps=30, bottom_depth=bottom_depth)
    return sum(c.push(kp, frame_id=f) is not None for f, kp in enumerate(seq)), c


@pytest.mark.parametrize('gap', [1, 9, 14, 15, 16, 20, 45])
def test_counter_recovers_after_gap(gap):
    seq = [stand()] * 30 + squats(2) + [missing()] * gap + [stand()] * 30 + squats(5)
    n, c = count(seq)
    assert n == 7, f'gap={gap}: counted {n}'
    assert c.state == 'READY'


@pytest.mark.parametrize('gap', [15, 16, 20])
def test_long_gap_recalibrates(gap):
    """At/after the reset threshold the old calibration must be dropped."""
    seq = [stand()] * 30 + [missing()] * gap + [stand(460)] * 30 + squats_from(460, 2)
    n, c = count(seq)
    assert n == 2 and c.stand_hip == pytest.approx(460, abs=2)


def test_pipeline_level_gap_15():
    pipe = SquatStreamPipeline(artifact=None, fps=30, bottom_depth=0.20, count_only=True)
    seq = [stand()] * 30 + squats(2) + [missing()] * 15 + [stand()] * 30 + squats(5)
    events = []
    for f, kp in enumerate(seq):
        events += pipe.push(kp, frame_id=f)
    assert len(events + pipe.flush()) == 7


def squats_from(hip, n):
    out = []
    for _ in range(n):
        down = [stand(hip + 200 * d) for d in [i * 0.6 / 19 for i in range(20)]]
        out += down + [stand(hip + 120)] * 5 + down[::-1] + [stand(hip)] * 15
    return out
