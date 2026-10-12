"""Online Squat repetition state machine driven by ordered YOLO17 keypoints.

No segmentation CSV, ground-truth boundaries, or labels are used. This is an
engineering starting point: thresholds MUST be validated and tuned on held-out
videos; pose calibration assumes an initially standing participant.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np

LH, RH, LK, RK, LA, RA, LS, RS = 11, 12, 13, 14, 15, 16, 5, 6


def point(kp, index, conf=0.30):
    item = kp[index]
    return item[:2].astype(float) if np.isfinite(item).all() and item[2] >= conf else None


def center(kp, a, b):
    x, y = point(kp, a), point(kp, b)
    if x is None and y is None:
        return None
    if x is None:
        return y
    if y is None:
        return x
    return (x + y) / 2


def joint_angle(a, b, c):
    if a is None or b is None or c is None:
        return float('nan')
    u, v = a - b, c - b
    den = np.linalg.norm(u) * np.linalg.norm(v)
    if den < 1e-6:
        return float('nan')
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / den, -1, 1))))


def measure_pose(kp):
    kp = np.asarray(kp, dtype=float)
    if kp.shape != (17, 3):
        raise ValueError('Expected [17,3] COCO keypoints')
    hip = center(kp, LH, RH)
    shoulder = center(kp, LS, RS)
    left = joint_angle(point(kp, LH), point(kp, LK), point(kp, LA))
    right = joint_angle(point(kp, RH), point(kp, RK), point(kp, RA))
    valid_angles = [x for x in (left, right) if np.isfinite(x)]
    knee = float(np.mean(valid_angles)) if valid_angles else float('nan')
    hy = float(hip[1]) if hip is not None else float('nan')
    torso = float(np.linalg.norm(hip - shoulder)) if (hip is not None and shoulder is not None) else float('nan')
    if not np.isfinite(torso) or torso < 10:
        torso = float('nan')
    return hy, knee, torso


def median_finite(items):
    ok = [x for x in items if np.isfinite(x)]
    return float(np.median(ok)) if ok else float('nan')


@dataclass
class CompletedRepetition:
    keypoints: np.ndarray
    start_frame: int
    end_frame: int
    peak_depth: float
    min_knee_angle: float


class SquatCounter:
    def __init__(self, fps=30.0, min_frames=15, calibration_frames=10,
                 start_depth=0.16, bottom_depth=0.45, finish_depth=0.13,
                 start_knee=145.0, bottom_knee=120.0, finish_knee=150.0):
        self.fps = float(fps)
        self.min_frames = min_frames
        self.calibration_frames = calibration_frames
        self.start_depth, self.bottom_depth = start_depth, bottom_depth
        self.finish_depth, self.start_knee = finish_depth, start_knee
        self.bottom_knee, self.finish_knee = bottom_knee, finish_knee
        self.state = 'CALIBRATING'
        self.frame_id = -1
        self.measurements = deque(maxlen=5)
        self.prebuffer = deque(maxlen=max(8, int(round(fps * .3))))
        self.calibration_y, self.calibration_scale = [], []
        self.stand_hip = float('nan')
        self.scale = float('nan')
        self.active = []
        self.start_frame = None
        self.start_votes = self.bottom_votes = self.finish_votes = 0
        self.peak_depth = 0.0
        self.min_knee = 180.0
        self.missing_count = 0
        self.count = 0
        self.max_frames = int(max(5, round(8 * fps)))

    def reset_rep(self):
        self.active = []
        self.start_frame = None
        self.start_votes = self.bottom_votes = self.finish_votes = 0
        self.peak_depth = 0.0
        self.min_knee = 180.0
        self.state = 'READY' if np.isfinite(self.stand_hip) else 'CALIBRATING'

    def push(self, kp, frame_id=None):
        """One frame at a time; optionally returns a completed repetition."""
        if frame_id is None:
            frame_id = self.frame_id + 1
        frame_id = int(frame_id)
        if frame_id <= self.frame_id:
            raise ValueError('Frame indices must increase')
        self.frame_id = frame_id
        kp = np.asarray(kp, dtype=np.float32)
        hy, knee, torso = measure_pose(kp)
        self.measurements.append((hy, knee, torso))
        sy = median_finite(m[0] for m in self.measurements)
        sk = median_finite(m[1] for m in self.measurements)
        ss = median_finite(m[2] for m in self.measurements)

        if not np.isfinite(hy):
            self.missing_count += 1
        else:
            self.missing_count = 0
        if self.missing_count >= max(10, int(self.fps * .5)):
            self.reset_rep()
            self.measurements.clear()
            self.prebuffer.clear()
            self.calibration_y.clear()
            self.calibration_scale.clear()
            self.stand_hip = float('nan')
            self.scale = float('nan')
            return None

        if self.state == 'CALIBRATING':
            # Calibration requires initially upright pose. 2D knee view may
            # distort this angle; calibration remains a test limitation.
            if np.isfinite(sy) and np.isfinite(ss) and (not np.isfinite(sk) or sk >= 145):
                self.calibration_y.append(sy)
                self.calibration_scale.append(ss)
                if len(self.calibration_y) >= self.calibration_frames:
                    self.stand_hip = float(np.median(self.calibration_y))
                    self.scale = float(np.median(self.calibration_scale))
                    self.state = 'READY'
            else:
                self.calibration_y.clear()
                self.calibration_scale.clear()
            self.prebuffer.append((frame_id, kp.copy()))
            return None

        if np.isfinite(ss):
            self.scale = .98 * self.scale + .02 * ss
        if not np.isfinite(sy) or not np.isfinite(self.scale):
            if self.state != 'READY':
                self.active.append((frame_id, kp.copy()))
            return None
        depth = (sy - self.stand_hip) / max(self.scale, 1e-6)
        moving = depth >= self.start_depth or (np.isfinite(sk) and sk <= self.start_knee)
        bottom = depth >= self.bottom_depth or (np.isfinite(sk) and sk <= self.bottom_knee)
        stand = depth <= self.finish_depth and (not np.isfinite(sk) or sk >= self.finish_knee)

        if self.state == 'READY':
            self.prebuffer.append((frame_id, kp.copy()))
            self.start_votes = self.start_votes + 1 if moving else 0
            if self.start_votes >= 3:
                self.active = list(self.prebuffer)
                self.start_frame = self.active[0][0]
                self.state = 'DESCENDING'
                self.peak_depth = max(0.0, depth)
                self.min_knee = sk if np.isfinite(sk) else 180.0
                self.bottom_votes = 0
                self.prebuffer.clear()
            elif not moving and abs(depth) < .07:
                self.stand_hip = .995 * self.stand_hip + .005 * sy
            return None

        self.active.append((frame_id, kp.copy()))
        self.peak_depth = max(self.peak_depth, depth)
        if np.isfinite(sk):
            self.min_knee = min(self.min_knee, sk)
        if len(self.active) > self.max_frames:
            self.reset_rep()
            return None

        if self.state == 'DESCENDING':
            self.bottom_votes = self.bottom_votes + 1 if bottom else 0
            if self.bottom_votes >= 3:
                self.state = 'BOTTOM'
                self.finish_votes = 0
            # A shallow motion is not counted.
            elif stand and len(self.active) > int(self.fps * 2):
                self.reset_rep()
        elif self.state == 'BOTTOM':
            if depth < self.peak_depth - .18 or (np.isfinite(sk) and sk >= self.min_knee + 15):
                self.state = 'ASCENDING'
        elif self.state == 'ASCENDING':
            self.finish_votes = self.finish_votes + 1 if stand else 0
            if self.finish_votes >= 4:
                frames = self.active
                peak, kneemin = self.peak_depth, self.min_knee
                self.reset_rep()
                if len(frames) >= self.min_frames:
                    self.count += 1
                    return CompletedRepetition(
                        keypoints=np.stack([p for _, p in frames]),
                        start_frame=frames[0][0], end_frame=frames[-1][0],
                        peak_depth=peak, min_knee_angle=kneemin)
        return None
