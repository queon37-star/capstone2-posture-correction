"""Measure gating cost on THIS machine (run it on the R2130-12; no camera needed).

Reports (1) wrapper overhead per frame with gating 0/1 and (2) the one-off
replay time at squat-session start, count-only and with the Good/Bad model.
Synthetic keypoints; the numbers are CPU cost, not recognition accuracy.

  python scripts/bench_auto_gating.py [--model models/squat_plus_asymmetry180_crop10_all9.joblib]
"""
import argparse
import contextlib
import io
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from auto_detect_hailo_bridge import AutoDetectPipeline          # noqa: E402
from auto_synthetic import FPS, ScriptedDetector, squats, stand  # noqa: E402
from squat_stream_pipeline import SquatStreamPipeline            # noqa: E402


def factory(artifact):
    return lambda: SquatStreamPipeline(artifact=artifact, fps=FPS, bottom_depth=0.20,
                                       count_only=artifact is None)


def per_frame_us(gating, artifact, seq, tmp):
    make = factory(artifact)
    with contextlib.redirect_stdout(io.StringIO()):
        p = AutoDetectPipeline(make(), FPS, events_path=Path(tmp) / f'f{gating}.jsonl',
                               kp_dump_path='', make_base=make, gating=gating)
        samples = []
        for f, kp in enumerate(seq):
            s = time.perf_counter()
            p.push(kp, frame_id=f, t=f / FPS)
            samples.append(1e6 * (time.perf_counter() - s))
        p.flush()
    return samples


def replay_ms(artifact, seq, tmp, repeats):
    make = factory(artifact)
    out = []
    for i in range(repeats):
        with contextlib.redirect_stdout(io.StringIO()):
            p = AutoDetectPipeline(make(), FPS, events_path=Path(tmp) / f'r{i}.jsonl',
                                   kp_dump_path='', make_base=make, gating=1,
                                   detector=ScriptedDetector([(0, 'idle'), (len(seq) - 1, 'squat')]))
            for f, kp in enumerate(seq):
                p.push(kp, frame_id=f, t=f / FPS)
        out.append((p.last_replay['ms'], p.last_replay['frames'], p.last_replay['reps']))
    return out


def summary(xs):
    xs = sorted(xs)
    return (f'mean={statistics.mean(xs):.0f} p50={xs[len(xs) // 2]:.0f} '
            f'p99={xs[int(.99 * (len(xs) - 1))]:.0f} max={xs[-1]:.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    artifacts = [('count_only', None)]
    if args.model:
        import joblib
        import squat_classification  # noqa: F401
        artifacts.append(('classifier', joblib.load(args.model)))
    seq = [stand()] * 30 + squats(30)           # ~60 s of frames
    with tempfile.TemporaryDirectory() as tmp:
        for name, artifact in artifacts:
            for g in (0, 1):
                print(f'{name:10s} gating={g} push() us/frame (incl. squat pipeline): '
                      + summary(per_frame_us(g, artifact, seq, tmp)))
            rows = replay_ms(artifact, [stand()] * 30 + squats(5), tmp, args.repeats)
            print(f'{name:10s} session-start replay: '
                  + ', '.join(f'{ms:.1f} ms ({fr} frames, {reps} reps emitted)' for ms, fr, reps in rows))
    print('Frame budget at 30 FPS: 33.3 ms per callback (Batch 4 delivers frames in groups).')


if __name__ == '__main__':
    main()
