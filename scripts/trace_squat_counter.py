#!/usr/bin/env python3
"""Trace online SquatCounter state on real video without modifying project code.
Run from ~/capstone2/pose_training; this script belongs in scripts/.
Only instrumentation: it monkey-patches SquatCounter.push in this process.
"""
import argparse
import runpy
import sys
from pathlib import Path

import numpy as np

scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))
from stream_squat_core import SquatCounter, measure_pose  # noqa: E402


def fmt(v):
    return f'{float(v):.2f}' if np.isfinite(v) else 'nan'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='PM_126')
    parser.add_argument('--max-frames', type=int, default=1250)
    parser.add_argument('--device', default='0')
    parser.add_argument('--trace-until', type=int, default=1250)
    parser.add_argument('--every', type=int, default=60)
    args = parser.parse_args()

    original_push = SquatCounter.push

    def traced_push(self, kp, frame_id=None):
        before = self.state
        result = original_push(self, kp, frame_id=frame_id)
        fid = self.frame_id
        if fid <= args.trace_until and (fid % args.every == 0 or before != self.state or result is not None):
            hy, knee, torso = measure_pose(kp)
            depth = ((hy - self.stand_hip) / self.scale
                     if np.isfinite(hy) and np.isfinite(self.stand_hip)
                     and np.isfinite(self.scale) and self.scale > 0 else np.nan)
            print(f'TRACE frame={fid:4d} state={before:11s}->{self.state:11s} '
                  f'hip_y={fmt(hy):>8s} knee={fmt(knee):>7s} torso={fmt(torso):>7s} '
                  f'depth={fmt(depth):>7s} cal={len(self.calibration_y):2d}/{self.calibration_frames} '
                  f'votes={self.start_votes},{self.bottom_votes},{self.finish_votes} '
                  f'missing={self.missing_count} reps={self.count}', flush=True)
        return result

    SquatCounter.push = traced_push
    runner = scripts_dir / 'run_squat_video_stream_buffered.py'
    if not runner.is_file():
        raise FileNotFoundError(f'Missing: {runner}')
    sys.argv = [str(runner), '--video', args.video, '--max-frames', str(args.max_frames),
                '--device', args.device,
                '--output', f'results/stream_test/{args.video}_trace_events.jsonl']
    print(f'Tracing {args.video}, first {args.max_frames} frames; no GT annotations loaded.', flush=True)
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()
