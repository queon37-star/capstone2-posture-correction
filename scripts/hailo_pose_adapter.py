"""Convert Hailo GStreamer detection landmarks to the existing COCO17 format.

Reference: hailo-ai/hailo-rpi5-examples/basic_pipelines/pose_estimation.py.
Hailo's landmark points are normalized relative to the *detection box*;
Hailo detection boxes are normalized relative to the callback video frame.

No `import hailo`: callers inject `hailo.HAILO_LANDMARKS` as `landmarks_type`,
so coordinate conversion can be unit-tested on a PC without HailoRT/Tappas.
"""

import numpy as np


COCO17_COUNT = 17
DEFAULT_CONFIDENCE_THRESHOLD = 0.30


def hailo_to_coco17(detection, width, height, *, landmarks_type,
                    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """Convert one Hailo person detection to float32 (17,3) pixel keypoints.

    Returns None if the detection has no landmark object. Assumes a compatible
    17-point YOLOv8-Pose COCO layout and the official GStreamer pose pipeline's
    normalized coordinate convention; this does not undo rotations, padding,
    cropping or rescaling outside that buffer.

    Callers must pass `width, height` from the callback pad's video caps,
    not the neural network's input tensor dimensions.
    """
    width, height = float(width), float(height)
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError('Video width and height must be finite and positive')
    if not (0 <= confidence_threshold <= 1):
        raise ValueError('confidence_threshold must be in [0, 1]')

    landmarks = detection.get_objects_typed(landmarks_type)
    if not landmarks:
        return None
    if len(landmarks) != 1:
        raise ValueError('Expected exactly one landmark object per person detection')

    points = landmarks[0].get_points()
    if len(points) != COCO17_COUNT:
        raise ValueError(f'Expected {COCO17_COUNT} COCO keypoints, got {len(points)}')

    bbox = detection.get_bbox()
    xmin, ymin = float(bbox.xmin()), float(bbox.ymin())
    box_width, box_height = float(bbox.width()), float(bbox.height())
    if (not np.isfinite([xmin, ymin, box_width, box_height]).all()
            or box_width <= 0 or box_height <= 0):
        raise ValueError('Invalid normalized Hailo detection bounding box')

    kp = np.zeros((COCO17_COUNT, 3), dtype=np.float32)
    kp[:, :2] = np.nan
    for i, point in enumerate(points):
        confidence = float(point.confidence())
        if not np.isfinite(confidence):
            confidence = 0.0
        kp[i, 2] = confidence
        if confidence < confidence_threshold:
            continue
        x, y = float(point.x()), float(point.y())
        if not np.isfinite([x, y]).all():
            continue
        kp[i, 0] = (x * box_width + xmin) * width
        kp[i, 1] = (y * box_height + ymin) * height
    return kp


def single_person_coco17(detections, width, height, *, landmarks_type,
                         confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """MVP: return pose only when *exactly one* person is detected.

    If no person or multiple people appear, return None rather than silently
    switching identities. The caller should feed `missing_pose()` into the
    squat pipeline for that frame, preserving consecutive frame indices.
    """
    persons = [det for det in detections if det.get_label() == 'person']
    if len(persons) != 1:
        return None
    return hailo_to_coco17(
        persons[0], width, height, landmarks_type=landmarks_type,
        confidence_threshold=confidence_threshold,
    )


def _normalized_bbox(detection):
    box = detection.get_bbox()
    return (float(box.xmin()), float(box.ymin()),
            float(box.width()), float(box.height()))


def _near_previous_bbox(previous, current):
    """Conservative single-person re-acquisition if a tracker ID changes."""
    if previous is None:
        return True
    x0, y0, w0, h0 = previous
    x1, y1, w1, h1 = current
    values = np.asarray([*previous, *current], dtype=float)
    if not np.isfinite(values).all() or min(w0, h0, w1, h1) <= 0:
        return False
    area_ratio = (w1 * h1) / (w0 * h0)
    return (abs((x1 + w1 / 2) - (x0 + w0 / 2)) <= 0.18
            and abs((y1 + h1 / 2) - (y0 + h0 / 2)) <= 0.30
            and 0.20 <= area_ratio <= 5.0)


class TrackedPersonSelector:
    """Select the same person by Hailo tracking ID in crowded frames.

    Acquire only from an unambiguous single-person frame. If that ID is
    absent, accept a replacement only when a single person has a compatible
    bbox near the previous subject. Never choose by highest confidence in
    a multi-person frame; do not silently switch identity on ambiguous data.
    """

    def __init__(self):
        self.active_track_id = None
        self.last_bbox = None
        self.multi_selected = 0
        self.reacquired_ids = 0

    @staticmethod
    def _track_id(detection, identity_type):
        if identity_type is None:
            return None
        identities = detection.get_objects_typed(identity_type)
        return int(identities[0].get_id()) if len(identities) == 1 else None

    def select(self, detections, width, height, *, landmarks_type,
               identity_type, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        persons = [det for det in detections if det.get_label() == 'person']
        if not persons:
            return None

        chosen = None
        if self.active_track_id is not None and identity_type is not None:
            matching = [
                det for det in persons
                if self._track_id(det, identity_type) == self.active_track_id
            ]
            if len(matching) == 1:
                chosen = matching[0]
            elif len(matching) > 1:
                return None  # Ambiguous duplicate tracker IDs.

        if chosen is None:
            if len(persons) != 1:
                return None  # Never guess between unknown people.
            candidate = persons[0]
            if not _near_previous_bbox(self.last_bbox, _normalized_bbox(candidate)):
                return None
            chosen = candidate

        kp = hailo_to_coco17(
            chosen, width, height, landmarks_type=landmarks_type,
            confidence_threshold=confidence_threshold,
        )
        if kp is None:
            return None

        chosen_id = self._track_id(chosen, identity_type)
        if (self.active_track_id is not None
                and chosen_id is not None and chosen_id != self.active_track_id):
            self.reacquired_ids += 1
        self.active_track_id = chosen_id
        self.last_bbox = _normalized_bbox(chosen)
        if len(persons) > 1:
            self.multi_selected += 1
        return kp
