"""Pure-Python fakes test Hailo landmark conversion without Hailo hardware."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from hailo_pose_adapter import hailo_to_coco17, single_person_coco17

LANDMARKS = object()


class Point:
    def __init__(self, x=0.5, y=0.25, confidence=0.95):
        self._x, self._y, self._confidence = x, y, confidence

    def x(self): return self._x
    def y(self): return self._y
    def confidence(self): return self._confidence


class BBox:
    def __init__(self, xmin=0.1, ymin=0.2, width=0.5, height=0.4):
        self._xmin, self._ymin = xmin, ymin
        self._width, self._height = width, height

    def xmin(self): return self._xmin
    def ymin(self): return self._ymin
    def width(self): return self._width
    def height(self): return self._height


class Landmarks:
    def __init__(self, points): self._points = points
    def get_points(self): return self._points


class Detection:
    def __init__(self, points=None, bbox=None, label='person'):
        self._points = points
        self._bbox = bbox or BBox()
        self._label = label

    def get_objects_typed(self, kind):
        assert kind is LANDMARKS
        return [] if self._points is None else [Landmarks(self._points)]

    def get_bbox(self): return self._bbox
    def get_label(self): return self._label


def test_bbox_relative_coordinates_to_frame_pixels():
    points = [Point() for _ in range(17)]
    out = hailo_to_coco17(Detection(points), 1280, 720, landmarks_type=LANDMARKS)
    assert out.shape == (17, 3) and out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], 448)
    np.testing.assert_allclose(out[:, 1], 216)
    np.testing.assert_allclose(out[:, 2], .95)
    # Output keeps fractions; it must not round to int like the overlay demo.
    points[11] = Point(.11111, .22222, .9)
    out = hailo_to_coco17(Detection(points), 1280, 720, landmarks_type=LANDMARKS)
    assert out[11, 0] == pytest.approx((.11111 * .5 + .1) * 1280)


def test_confidence_mask_and_nan_coordinate():
    points = [Point() for _ in range(17)]
    points[0] = Point(.1, .2, .29)
    points[1] = Point(.1, .2, .30)
    points[2] = Point(np.nan, .2, .99)
    points[3] = Point(.1, .2, np.nan)
    out = hailo_to_coco17(Detection(points), 640, 480, landmarks_type=LANDMARKS)
    assert np.isnan(out[0, :2]).all() and out[0, 2] == pytest.approx(.29)
    assert np.isfinite(out[1, :2]).all()
    assert np.isnan(out[2, :2]).all() and out[2, 2] == pytest.approx(.99)
    assert np.isnan(out[3, :2]).all() and out[3, 2] == 0


def test_no_landmarks_and_wrong_point_count():
    assert hailo_to_coco17(Detection(points=None), 640, 480,
                           landmarks_type=LANDMARKS) is None
    with pytest.raises(ValueError, match='17'):
        hailo_to_coco17(Detection([Point()] * 16), 640, 480,
                        landmarks_type=LANDMARKS)


def test_single_person_policy():
    person = Detection([Point()] * 17)
    other = Detection([Point()] * 17, label='dog')
    assert single_person_coco17([], 640, 480, landmarks_type=LANDMARKS) is None
    assert single_person_coco17([person, person], 640, 480,
                                landmarks_type=LANDMARKS) is None
    assert single_person_coco17([person, other], 640, 480,
                                landmarks_type=LANDMARKS).shape == (17, 3)
    assert single_person_coco17([Detection()], 640, 480,
                                landmarks_type=LANDMARKS) is None


def test_invalid_dims_and_bbox():
    person = Detection([Point()] * 17)
    with pytest.raises(ValueError, match='width and height'):
        hailo_to_coco17(person, 0, 480, landmarks_type=LANDMARKS)
    with pytest.raises(ValueError, match='bounding box'):
        hailo_to_coco17(Detection([Point()] * 17, BBox(width=0)),
                        640, 480, landmarks_type=LANDMARKS)
    with pytest.raises(ValueError, match='confidence_threshold'):
        hailo_to_coco17(person, 640, 480, landmarks_type=LANDMARKS,
                        confidence_threshold=-0.1)


def test_coco_knee_hip_shoulder_indices_are_preserved():
    width, height = 1280, 720
    # Pose in pixel coordinates chosen to match the existing counter test.
    xy = {5: (475, 240), 6: (525, 240),
          11: (475, 400), 12: (525, 400),
          13: (475, 510), 14: (525, 510),
          15: (477, 620), 16: (527, 620)}
    points = [Point(0.0, 0.0, confidence=0.0) for _ in range(17)]
    for index, (x, y) in xy.items():
        points[index] = Point(x / width, y / height, confidence=.98)
    full_frame = BBox(xmin=0, ymin=0, width=1, height=1)
    kp = hailo_to_coco17(Detection(points, bbox=full_frame),
                         width, height, landmarks_type=LANDMARKS)
    assert kp[11, 1] == pytest.approx(400)  # left hip
    assert kp[12, 1] == pytest.approx(400)  # right hip
    assert kp[5, 1] == pytest.approx(240)   # left shoulder
    assert kp[6, 1] == pytest.approx(240)   # right shoulder
    assert kp[13, 1] == pytest.approx(510)  # left knee
    assert kp[14, 1] == pytest.approx(510)  # right knee
    assert kp[15, 1] == pytest.approx(620)  # left ankle
    assert kp[16, 1] == pytest.approx(620)  # right ankle
    assert np.isnan(kp[0, :2]).all()
