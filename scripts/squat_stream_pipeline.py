"""Pose-backend-agnostic, frame-by-frame squat analysis.

Supply one selected person's COCO17 [x_pixels, y_pixels, confidence]
array (shape 17x3, dtype convertible to float32) for *every* source frame.
This module never runs a Pose model or reads video/ground truth.
"""

import numpy as np

from squat_rep_buffer import RepBoundaryBuffer
from stream_squat_core import SquatCounter


def missing_pose():
    """Use when no subject is detected; preserves original missing-data semantics."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, :2] = np.nan
    return kp


class SquatStreamPipeline:
    """Count, buffer and classify squat repetitions from sequential keypoints.

    The returned list contains zero or more 'rep' dictionaries. It may be
    empty even when a repetition was just counted, because 'post_frames'
    future observations are needed to classify it. Pass the dictionaries to
    a JSONL writer or transport layer outside this class.
    """

    def __init__(self, *, artifact, fps, bottom_depth=0.45,
                 bottom_knee=120.0, pre_frames=16, post_frames=22,
                 classifier=None, count_only=False):
        self.artifact = artifact
        self.classifier = classifier
        self.count_only = bool(count_only)
        self.counter = SquatCounter(fps=fps, bottom_depth=bottom_depth,
                                    bottom_knee=bottom_knee)
        self.assembler = RepBoundaryBuffer(fps=fps, pre_frames=pre_frames,
                                            post_frames=post_frames)
        self.event_count = 0

    def push(self, keypoints, frame_id):
        """Process one frame. Call even when Pose fails (use missing_pose())."""
        kp = np.asarray(keypoints, dtype=np.float32)
        if kp.shape != (17, 3):
            raise ValueError('Expected (17, 3) COCO keypoints')
        rep_event = self.counter.push(kp, frame_id=frame_id)
        return [self._make_event(rep)
                for rep in self.assembler.push(frame_id, kp, rep_event)]

    def flush(self):
        """Finalize pending reps at end-of-stream, including partial post-context."""
        return [self._make_event(rep) for rep in self.assembler.flush()]

    def _make_event(self, completed_rep):
        self.event_count += 1
        if self.count_only:
            # Hardware integration smoke test: no classifier/model dependency.
            message = {
                'type': 'rep', 'exercise': 'squat', 'count': self.event_count,
                'posture': 'unknown', 'issues': [], 'score': None,
                'classification_status': 'disabled_count_only',
            }
        else:
            try:
                if self.classifier is None:
                    from squat_classification import classify
                    classify_fn = classify
                else:
                    classify_fn = self.classifier
                pred, prob = classify_fn(completed_rep.keypoints, self.artifact)
                message = {
                    'type': 'rep', 'exercise': 'squat', 'count': self.event_count,
                    'posture': 'good' if pred == 1 else 'bad',
                    'issues': [], 'score': None,
                    'probability_correct_uncalibrated': round(prob, 4),
                }
            except Exception as exc:
                message = {
                    'type': 'rep', 'exercise': 'squat', 'count': self.event_count,
                    'posture': 'unknown', 'issues': [], 'score': None,
                    'classification_error': str(exc),
                }
        message.update({
            'start_frame': completed_rep.start_frame,
            'end_frame': completed_rep.end_frame,
            'detected_start_frame': completed_rep.detected_start_frame,
            'detected_end_frame': completed_rep.detected_end_frame,
            'classification_frame_count': len(completed_rep.keypoints),
            'post_context_complete': completed_rep.post_context_complete,
            'peak_hip_depth_torso_units': round(completed_rep.peak_depth, 3),
            'min_projected_knee_angle': round(completed_rep.min_knee_angle, 1),
        })
        return message
