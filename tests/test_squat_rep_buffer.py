"""Unit tests for boundary buffering; no GPU or dataset required."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from squat_rep_buffer import RepBoundaryBuffer
from dataclasses import dataclass

@dataclass
class CompletedRepetition:
    keypoints: np.ndarray
    start_frame: int
    end_frame: int
    peak_depth: float
    min_knee_angle: float


def kp(f):
    return np.full((17, 3), float(f), np.float32)


def rep(start, end):
    return CompletedRepetition(keypoints=np.stack([kp(i) for i in range(start, end + 1)]),
                               start_frame=start, end_frame=end,
                               peak_depth=0.8, min_knee_angle=98.0)


def test_padding_and_delay():
    b = RepBoundaryBuffer(fps=30, pre_frames=4, post_frames=5)
    emitted = []
    for i in range(50):
        emitted += b.push(i, kp(i), rep(10, 30) if i == 30 else None)
        if i < 35:
            assert not emitted
    assert len(emitted) == 1
    v = emitted[0]
    assert (v.start_frame, v.end_frame) == (6, 35)
    assert len(v.keypoints) == 30 and v.post_context_complete
    assert np.array_equal(v.keypoints[:, 0, 0], np.arange(6, 36))


def test_no_overlap():
    b = RepBoundaryBuffer(fps=30, pre_frames=10, post_frames=6)
    emitted = []
    for i in range(90):
        event = rep(20, 40) if i == 40 else rep(50, 70) if i == 70 else None
        emitted += b.push(i, kp(i), event)
    assert len(emitted) == 2
    assert emitted[0].end_frame == 46
    assert emitted[1].start_frame == 47
    assert emitted[1].end_frame == 76


def test_flush_incomplete_post_context():
    b = RepBoundaryBuffer(fps=30, pre_frames=3, post_frames=8)
    for i in range(32):
        assert not b.push(i, kp(i), rep(12, 30) if i == 30 else None)
    emitted = b.flush()
    assert len(emitted) == 1
    assert emitted[0].start_frame == 9
    assert emitted[0].end_frame == 31
    assert not emitted[0].post_context_complete


if __name__ == '__main__':
    test_padding_and_delay()
    test_no_overlap()
    test_flush_incomplete_post_context()
    print('Buffered repetition boundary tests: PASS')
