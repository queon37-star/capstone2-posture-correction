"""Synthetic COCO17 sequences and a scripted detector for AutoDetectPipeline tests.

These are toy kinematics (see tests/test_squat_depth_threshold.py). Passing
tests here say nothing about real exercise-recognition accuracy.
"""
import numpy as np

FPS = 30.0


def missing():
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, :2] = np.nan
    return kp


def stand(hip_y=400.0, x0=500.0):
    """Upright person; hip_y > 400 means the hips moved down (squat)."""
    kp = missing()
    pts = {0: (x0, hip_y - 260), 5: (x0 - 25, hip_y - 200), 6: (x0 + 25, hip_y - 200),
           9: (x0 - 30, hip_y - 60), 10: (x0 + 30, hip_y - 60),
           11: (x0 - 25, hip_y), 12: (x0 + 25, hip_y),
           13: (x0 - 25, hip_y + 100), 14: (x0 + 25, hip_y + 100),
           15: (x0 - 25, hip_y + 210), 16: (x0 + 25, hip_y + 210)}
    for i, (x, y) in pts.items():
        kp[i] = [x, y, 0.98]
    return kp


def plank(shoulder_y=500.0):
    """Side-view push-up: the body rotates about the ankles on the floor."""
    ankle = np.array([1000.0, 690.0])
    sh = np.array([450.0, shoulder_y])
    hip = ankle + 0.64 * (sh - ankle)
    knee = ankle + 0.32 * (sh - ankle)
    wrist = np.array([450.0, 690.0])
    kp = missing()
    for i, p in {0: sh + [-60, -10], 5: sh, 6: sh + [4, 0], 9: wrist, 10: wrist + [4, 0],
                 11: hip, 12: hip + [4, 0], 13: knee, 14: knee + [4, 0],
                 15: ankle, 16: ankle + [4, 0]}.items():
        kp[i] = [p[0], p[1], 0.95]
    return kp


def squat_rep(depth=0.6):
    """One squat: 20 down, 5 hold, 20 up, 15 standing (60 frames, 2 s)."""
    down = [stand(400 + 200 * d) for d in np.linspace(0, depth, 20)]
    up = [stand(400 + 200 * d) for d in np.linspace(depth, 0, 20)]
    return down + [stand(400 + 200 * depth)] * 5 + up + [stand()] * 15


def squats(n, depth=0.6):
    out = []
    for _ in range(n):
        out += squat_rep(depth)
    return out


def pushups(n, up=500.0, down=640.0):
    out = []
    for _ in range(n):
        out += ([plank(y) for y in np.linspace(up, down, 20)] + [plank(down)] * 5
                + [plank(y) for y in np.linspace(down, up, 20)] + [plank(up)] * 10)
    return out


class ScriptedDetector:
    """Stand-in for ExerciseDetector: returns a confirmed label per frame from a
    list of (first_frame, label) changes, so gating tests do not depend on the
    heuristic thresholds. reset() is recorded but does not alter the script."""

    def __init__(self, changes, fps=FPS):
        self.changes = sorted(changes)
        self.fps = fps
        self.last_raw = None
        self.last_info = {}
        self.buf = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def update(self, kp, t):
        frame = int(round(t * self.fps))
        label = None
        for start, value in self.changes:
            if frame >= start:
                label = value
        self.last_raw = label
        return label
