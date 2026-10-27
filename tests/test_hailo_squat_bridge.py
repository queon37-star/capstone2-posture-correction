"""Bridge tests without HailoRT, GStreamer, a real camera or a trained model."""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from hailo_squat_bridge import HailoSquatBridge


LANDMARKS = object()


class Point:
    def x(self): return 0.5
    def y(self): return 0.25
    def confidence(self): return 0.95


class BBox:
    def xmin(self): return 0.1
    def ymin(self): return 0.2
    def width(self): return 0.5
    def height(self): return 0.4


class Landmarks:
    def get_points(self): return [Point() for _ in range(17)]


class Detection:
    def __init__(self, label='person', has_pose=True):
        self.label, self.has_pose = label, has_pose

    def get_label(self): return self.label
    def get_bbox(self): return BBox()

    def get_objects_typed(self, kind):
        assert kind is LANDMARKS
        return [Landmarks()] if self.has_pose else []


class FakePipeline:
    def __init__(self):
        self.seen = []
        self.flushed = False

    def push(self, keypoints, frame_id):
        self.seen.append((frame_id, keypoints.copy()))
        return [{'type': 'rep', 'count': 1}] if frame_id == 2 else []

    def flush(self):
        self.flushed = True
        return [{'type': 'rep', 'count': 2}]


def test_frame_by_frame_bridge_and_jsonl():
    pipeline = FakePipeline()
    writer = io.StringIO()
    bridge = HailoSquatBridge(pipeline, writer)

    bridge.process(frame_id=0, detections=[Detection()], width=1280,
                   height=720, landmarks_type=LANDMARKS)
    bridge.process(frame_id=1, detections=[Detection(), Detection()],
                   width=1280, height=720, landmarks_type=LANDMARKS)
    bridge.process(frame_id=2, detections=[], width=1280,
                   height=720, landmarks_type=LANDMARKS)
    bridge.finish()

    assert [fid for fid, _ in pipeline.seen] == [0, 1, 2]
    np.testing.assert_allclose(pipeline.seen[0][1][:, 0], 448)
    np.testing.assert_allclose(pipeline.seen[0][1][:, 1], 216)
    assert np.isnan(pipeline.seen[1][1][:, :2]).all()
    assert np.isnan(pipeline.seen[2][1][:, :2]).all()
    assert pipeline.seen[1][1][:, 2].sum() == 0
    assert (bridge.processed_frames, bridge.valid_person_frames,
            bridge.missing_person_frames) == (3, 1, 2)
    assert pipeline.flushed
    assert [json.loads(line)['count'] for line in writer.getvalue().splitlines()] == [1, 2]


def test_missing_caps_or_landmarks_are_missing_pose_not_dropped_frames():
    pipeline = FakePipeline()
    writer = io.StringIO()
    bridge = HailoSquatBridge(pipeline, writer)
    bridge.process(frame_id=0, detections=[Detection()], width=None,
                   height=None, landmarks_type=LANDMARKS)
    bridge.process(frame_id=1, detections=[Detection(has_pose=False)],
                   width=1280, height=720, landmarks_type=LANDMARKS)
    assert bridge.processed_frames == bridge.missing_person_frames == 2
    assert bridge.valid_person_frames == 0
    assert all(np.isnan(kp[:, :2]).all() for _, kp in pipeline.seen)
