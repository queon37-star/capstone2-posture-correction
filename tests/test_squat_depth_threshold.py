"""Synthetic comparison of depth thresholds; no video or GT accessed.

Run from project root: python tests/test_squat_depth_threshold.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stream_squat_core import SquatCounter


def pose_at_depth(depth):
    """Toy kinematics; knee appears almost straight, hip moves down."""
    hip_y = 400 + 200 * depth
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, :2] = np.nan
    for i, (x, y) in {
        5: (475, hip_y - 200), 6: (525, hip_y - 200),
        11: (475, hip_y), 12: (525, hip_y),
        13: (475, hip_y + 100), 14: (525, hip_y + 100),
        15: (475, hip_y + 210), 16: (525, hip_y + 210),
    }.items():
        kp[i] = [x, y, .98]
    return kp


def run(profile, bottom_depth):
    c = SquatCounter(fps=30, bottom_depth=bottom_depth)
    events = []
    for fid, value in enumerate(profile):
        event = c.push(pose_at_depth(value), frame_id=fid)
        if event is not None:
            events.append(event)
    return events


def rep_profile(peak):
    return [0] * 24 + np.linspace(0, peak, 20).tolist() + [peak] * 8 + np.linspace(peak, 0, 20).tolist() + [0] * 15


def main():
    profile = [0] * 30
    peaks = (.28, .30, .33, .28, .22, .31)
    for p in peaks:
        profile.extend(rep_profile(p))
    old = run(profile, .45)
    lowered = run(profile, .20)
    assert len(old) == 0, f'Unexpected original events: {len(old)}'
    assert len(lowered) == len(peaks), f'Expected {len(peaks)} shallow reps, got {len(lowered)}'
    quiet = [0] * 30 + (.04 + .03 * np.sin(np.linspace(0, 20, 300))).tolist()
    assert not run(quiet, .20), 'Stationary low-amplitude motion must not count'
    print('PASS: original 0.45 misses shallow synthetic squats; 0.20 recognizes them')
    print('PASS: low-amplitude stationary movements not counted')
    print('Caution: this synthetic test is not a real-video false-positive evaluation.')


if __name__ == '__main__':
    main()
