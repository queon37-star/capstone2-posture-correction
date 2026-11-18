"""Replay a SQUAT_KP_DUMP recording offline through the same squat/auto pipeline.

Same keypoints, same frame ids and the same detector time values as the board
run, so gating OFF/ON (or any later change) can be compared without a camera.

  python scripts/replay_kp_dump.py results/hailo/kp_20261119.bin --gating both
  python scripts/replay_kp_dump.py kp.bin --gating 1 --model models/squat_plus_asymmetry180_crop10_all9.joblib
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

from auto_detect_hailo_bridge import AutoDetectPipeline, read_kp_dump
from squat_stream_pipeline import SquatStreamPipeline


def replay(data, *, gating, fps, out_dir, artifact, bottom_depth, quiet):
    def make_base():
        # Same thresholds as run_squat_hailo_pose.py main().
        return SquatStreamPipeline(artifact=artifact, fps=fps, bottom_depth=bottom_depth,
                                   bottom_knee=120.0, pre_frames=16, post_frames=22,
                                   count_only=artifact is None)

    tag = f'gating{gating}'
    reps = []
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        p = AutoDetectPipeline(make_base(), fps, events_path=out_dir / f'{tag}_events.jsonl',
                               kp_dump_path='', make_base=make_base, gating=gating)
        for row in data:
            reps += p.push(row['kp'], frame_id=int(row['frame_id']), t=float(row['t']))
        reps += p.flush()
    with (out_dir / f'{tag}_reps.jsonl').open('w', encoding='utf-8') as f:
        for r in reps:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return reps, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump')
    ap.add_argument('--gating', choices=['0', '1', 'both'], default='both')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--model', default=None, help='joblib artifact; omit for count-only')
    ap.add_argument('--bottom-depth', type=float, default=0.20)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    dump = Path(args.dump)
    meta_path = dump.with_name(dump.name + '.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
    fps = float(meta.get('assumed_fps', 30.0))
    data = read_kp_dump(dump)
    out_dir = Path(args.out_dir) if args.out_dir else dump.with_name(dump.stem + '_replay')
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = None
    if args.model:
        import joblib
        import squat_classification  # noqa: F401  (import cost outside the loop)
        artifact = joblib.load(args.model)
    print(f'frames={len(data)} pose_invalid={int((data["pose_valid"] == 0).sum())} '
          f'fps={fps:g} timebase={meta.get("timebase")} out={out_dir}')
    for g in ([0, 1] if args.gating == 'both' else [int(args.gating)]):
        reps, p = replay(data, gating=g, fps=fps, out_dir=out_dir, artifact=artifact,
                         bottom_depth=args.bottom_depth, quiet=not args.verbose)
        print(f'gating={g}: reps={len(reps)} '
              + ' '.join(f'[{r["detected_start_frame"]}-{r["detected_end_frame"]}'
                         f'{" s" + str(r["session_id"]) if "session_id" in r else ""}'
                         f' {r.get("posture")}]' for r in reps))
        if g:
            print(f'  max_replay_ms={p.max_replay_ms:.1f} sessions={p.session_id}')


if __name__ == '__main__':
    main()
