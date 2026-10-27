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
