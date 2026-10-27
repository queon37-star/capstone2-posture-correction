"""Backend-independent squat stream regression checks (no GPU or video needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from squat_stream_pipeline import SquatStreamPipeline, missing_pose


def pose(hip_y):
    kp = missing_pose()
    for idx, (x, y) in {
        5: (475, hip_y - 160), 6: (525, hip_y - 160),
        11: (475, hip_y), 12: (525, hip_y),
        13: (475, hip_y + 110), 14: (525, hip_y + 110),
        15: (477, hip_y + 220), 16: (527, hip_y + 220),
    }.items():
        kp[idx] = [x, y, 0.98]
    return kp


def squat_frames():
    # Stand to calibrate -> descend -> bottom -> ascend -> stand.
    return [pose(y) for y in ([400] * 15 + [430] * 5 + [485] * 7
                              + [435] * 5 + [400] * 8)]


def test_one_repetition_with_post_context():
    frames = squat_frames() + [pose(400)] * 28
    seen = []

    def fake_classifier(seq, artifact):
        assert artifact == 'model'
        seen.append(seq.copy())
        return 1, 0.87456

    pipeline = SquatStreamPipeline(
        artifact='model', fps=30, bottom_depth=0.20,
        pre_frames=16, post_frames=22, classifier=fake_classifier)
    events = []
    for i, kp in enumerate(frames):
        events.extend(pipeline.push(kp, frame_id=i))
    events.extend(pipeline.flush())

    assert pipeline.counter.count == 1
    assert pipeline.event_count == 1
    assert len(seen) == len(events) == 1
    event = events[0]
    assert (event['type'], event['exercise'], event['count']) == ('rep', 'squat', 1)
    assert (event['posture'], event['issues'], event['score']) == ('good', [], None)
    assert event['probability_correct_uncalibrated'] == 0.8746
    assert event['post_context_complete'] is True
    assert event['classification_frame_count'] == len(seen[0])
    assert event['start_frame'] <= event['detected_start_frame']
    assert event['end_frame'] >= event['detected_end_frame']


def test_flush_preserves_partial_post_context_and_unknown():
    frames = squat_frames()

    def broken_classifier(seq, artifact):
        raise ValueError('unavailable feature')

    pipeline = SquatStreamPipeline(
        artifact=None, fps=30, bottom_depth=0.20,
        pre_frames=16, post_frames=22, classifier=broken_classifier)
    events = []
    for i, kp in enumerate(frames):
        events.extend(pipeline.push(kp, frame_id=i))
    assert not events
    events = pipeline.flush()
    assert len(events) == 1
    assert events[0]['posture'] == 'unknown'
    assert events[0]['classification_error'] == 'unavailable feature'
    assert events[0]['post_context_complete'] is False
    assert events[0]['count'] == 1


def test_bad_pose_shape_and_missing_pose():
    pipeline = SquatStreamPipeline(
        artifact=None, fps=30, classifier=lambda *_: (1, 0.5))
    empty = missing_pose()
    assert empty.shape == (17, 3) and empty.dtype == np.float32
    assert np.isnan(empty[:, :2]).all() and (empty[:, 2] == 0).all()
    assert pipeline.push(empty, frame_id=0) == []
    with pytest.raises(ValueError, match='17, 3'):
        pipeline.push(np.zeros((16, 3)), frame_id=1)
