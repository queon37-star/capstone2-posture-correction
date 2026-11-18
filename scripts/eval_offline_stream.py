"""Offline stream evaluation on full-video keypoint caches (PC / WSL only).

Replays each cache through the UNCHANGED SquatCounter / SquatStreamPipeline and
compares, on identical keypoints:
  buffer policy : no_overlap (current RepBoundaryBuffer) | overlap (windows of
                  consecutive reps may share frames; experiment-only subclass)
  gating        : off | on   (AutoDetectPipeline, SQUAT_AUTO_GATING semantics)
Good/Bad uses the person-held-out model of each video (train_lopo_heldout_models.py).

Matching and scoring (fixes two issues of evaluate_squat_stream.py):
  * GT reps are matched to the DETECTED interval (detected_start..detected_end),
    1:1 by IoU >= --min-iou. The classifier input window (start..end incl.
    pre/post context) is reported separately as coverage, never used to match.
  * posture=unknown is a classification FAILURE, never a correct answer.
    accuracy_classified = correct / (good|bad predictions on matched reps)
    failure_rate        = unknown / matched
    accuracy_end_to_end = correct / all GT reps (missed + unknown count as wrong)
  * The whole cache is evaluated by default (no 1199-frame cut-off).

  python scripts/eval_offline_stream.py --models-dir models/lopo_heldout/PlusAsymmetry180_crop10
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_detect_hailo_bridge import AutoDetectPipeline, read_kp_dump  # noqa: E402
from squat_rep_buffer import RepBoundaryBuffer  # noqa: E402
from squat_stream_pipeline import SquatStreamPipeline  # noqa: E402
from stream_squat_core import LH, RH, point  # noqa: E402

CONDITIONS = [('no_overlap', 'off'), ('overlap', 'off'), ('no_overlap', 'on'), ('overlap', 'on')]


class OverlapRepBoundaryBuffer(RepBoundaryBuffer):
    """Experiment-only policy: each rep keeps its full pre/post context even if it
    overlaps the previous rep's window. Everything else is the original code."""

    def _finalize(self, event, available_through, post_context_complete):
        self.previous_emitted_end = -1
        return super()._finalize(event, available_through, post_context_complete)


# ----------------------------------------------------------------------- inputs
def load_cache(path):
    path = Path(path)
    meta_path = path.with_name(path.name + '.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
    data = read_kp_dump(path)
    if len(data) and not np.array_equal(data['frame_id'], np.arange(data['frame_id'][0],
                                                                    data['frame_id'][0] + len(data))):
        raise ValueError(f'{path}: frame ids are not contiguous')
    return data, meta


def load_gt(segmentation_csv, video_id, exercise_id='6'):
    with open(segmentation_csv, newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f, delimiter=';')
                if r['video_id'] == video_id and r['exercise_id'] == exercise_id]
    return sorted(({'rep': int(r['repetition_number']), 'start': int(r['first_frame']),
                    'end': int(r['last_frame']), 'label': int(r['correctness']),
                    'person_id': int(r['person_id'])} for r in rows), key=lambda g: g['start'])


def mask_conf(kp, threshold):
    if threshold is None or threshold <= 0:
        return kp
    kp = kp.copy()
    kp[kp[:, 2] < threshold, :2] = np.nan
    return kp


def exact_gap_runs(kps, length=15, conf=0.30):
    """Count hip-missing runs of exactly `length` frames (the 0003 counter fix
    changes behaviour only for these)."""
    runs, current, out = [], 0, 0
    for kp in kps:
        missing = point(kp, LH, conf) is None and point(kp, RH, conf) is None
        if missing:
            current += 1
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return sum(1 for r in runs if r == length)


# ------------------------------------------------------------------ replaying
def make_factory(artifact, fps, policy, classifier, *, bottom_depth, bottom_knee, pre, post):
    def make():
        pipe = SquatStreamPipeline(artifact=artifact, fps=fps, bottom_depth=bottom_depth,
                                   bottom_knee=bottom_knee, pre_frames=pre, post_frames=post,
                                   classifier=classifier, count_only=artifact is None)
        if policy == 'overlap':
            pipe.assembler = OverlapRepBoundaryBuffer(fps=fps, pre_frames=pre, post_frames=post)
        elif policy != 'no_overlap':
            raise ValueError(policy)
        return pipe
    return make


def replay(data, fps, factory, gating, events_path, mask, log_path=None):
    reps = []
    log = open(log_path, 'w', encoding='utf-8') if log_path else open(Path(events_path).with_suffix('.log'), 'w')
    with log, contextlib.redirect_stdout(log):   # AUTO_* lines go to a per-run log
        p = AutoDetectPipeline(factory(), fps, events_path=events_path, kp_dump_path='',
                               make_base=factory, gating=1 if gating == 'on' else 0)
        started = time.perf_counter()
        for row in data:
            reps += p.push(mask_conf(row['kp'], mask), frame_id=int(row['frame_id']), t=float(row['t']))
        reps += p.flush()
        elapsed = time.perf_counter() - started
    events = [json.loads(line) for line in Path(events_path).read_text(encoding='utf-8').splitlines()]
    return reps, events, {'seconds': round(elapsed, 3), 'max_replay_ms': round(p.max_replay_ms, 2)}


# ------------------------------------------------------------------- scoring
def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def match(gt, reps, min_iou):
    """Greedy 1:1 by IoU on DETECTED intervals (same greedy rule as before)."""
    cands = sorted(((iou((g['start'], g['end']), (r['detected_start_frame'], r['detected_end_frame'])), i, j)
                    for i, g in enumerate(gt) for j, r in enumerate(reps)), reverse=True)
    used_g, used_r, pairs = set(), set(), {}
    for value, i, j in cands:
        if value < min_iou:
            break
        if i in used_g or j in used_r:
            continue
        used_g.add(i)
        used_r.add(j)
        pairs[i] = (j, value)
    return pairs


def in_range(start, end, lo, hi):
    return start >= lo and end <= hi


def evaluate_video(video, gt_all, reps_all, *, lo, hi, min_iou, condition,
                   first_confirm=None, session_confirms=None):
    """Evaluate matched repetitions; session_confirms maps session_id to confirmation frame.

    first_confirm is retained for legacy single-session callers only.
    """
    gt = [g for g in gt_all if in_range(g['start'], g['end'], lo, hi)]
    reps = [r for r in reps_all if in_range(r['detected_start_frame'], r['detected_end_frame'], lo, hi)]
    pairs = match(gt, reps, min_iou)
    rows = []
    for i, g in enumerate(gt):
        row = {'condition': condition, 'video_id': video, 'person_id': g['person_id'], 'gt_rep': g['rep'],
               'gt_start': g['start'], 'gt_end': g['end'], 'gt_label': g['label'], 'matched': i in pairs}
        if i in pairs:
            j, value = pairs[i]
            r = reps[j]
            win = (r['start_frame'], r['end_frame'])
            inter = max(0, min(win[1], g['end']) - max(win[0], g['start']) + 1)
            posture = r.get('posture', 'unknown')
            if session_confirms is not None:
                confirm_frame = session_confirms.get(r.get('session_id'))
            else:
                confirm_frame = first_confirm
            row.update({
                'det_start': r['detected_start_frame'], 'det_end': r['detected_end_frame'],
                'det_iou': round(value, 4),
                'win_start': win[0], 'win_end': win[1], 'classification_frames': r.get('classification_frame_count'),
                'pre_context_frames': r['detected_start_frame'] - r['start_frame'],
                'window_gt_coverage': round(inter / (g['end'] - g['start'] + 1), 4),
                'window_iou': round(iou(win, (g['start'], g['end'])), 4),
                'posture': posture, 'classified': posture in ('good', 'bad'),
                'correct': (posture == 'good' and g['label'] == 1) or (posture == 'bad' and g['label'] == 0),
                'prob_correct': r.get('probability_correct_uncalibrated'),
                'classification_error': r.get('classification_error'),
                'session_id': r.get('session_id'),
                'session_confirm_frame': confirm_frame,
                'before_session_confirm': (r['detected_end_frame'] < confirm_frame
                                           if confirm_frame is not None else None)})
        else:
            row.update({'posture': 'missed', 'classified': False, 'correct': False})
        rows.append(row)
    extras = [{'condition': condition, 'video_id': video, 'extra': True,
               'det_start': r['detected_start_frame'], 'det_end': r['detected_end_frame'],
               'posture': r.get('posture', 'unknown')}
              for j, r in enumerate(reps) if j not in {jj for jj, _ in pairs.values()}]
    excluded = len(gt_all) - len(gt)
    return rows, extras, excluded


def summarize(rows, extras):
    gt = len(rows)
    m = [r for r in rows if r['matched']]
    cls = [r for r in m if r['classified']]
    correct = sum(r['correct'] for r in cls)
    bad_gt = [r for r in rows if r['gt_label'] == 0]
    tp_bad = sum(1 for r in cls if r['gt_label'] == 0 and r['posture'] == 'bad')
    fn_bad = sum(1 for r in cls if r['gt_label'] == 0 and r['posture'] == 'good')
    fp_bad = sum(1 for r in cls if r['gt_label'] == 1 and r['posture'] == 'bad')
    tn = sum(1 for r in cls if r['gt_label'] == 1 and r['posture'] == 'good')
    pre = [r['pre_context_frames'] for r in m]

    def ratio(a, b):
        return round(a / b, 4) if b else None

    rec_bad, rec_good = ratio(tp_bad, tp_bad + fn_bad), ratio(tn, tn + fp_bad)
    return {
        'gt_reps': gt, 'detected_matched': len(m), 'missed': gt - len(m), 'extra_detections': len(extras),
        'mean_det_iou': round(statistics.mean(r['det_iou'] for r in m), 4) if m else None,
        'classified': len(cls), 'classification_failed': len(m) - len(cls),
        'failure_rate': ratio(len(m) - len(cls), len(m)),
        'accuracy_classified': ratio(correct, len(cls)),
        'accuracy_matched_unknown_wrong': ratio(correct, len(m)),
        'accuracy_end_to_end': ratio(correct, gt),
        'bad_recall_classified': rec_bad, 'bad_precision_classified': ratio(tp_bad, tp_bad + fp_bad),
        'bad_recall_end_to_end': ratio(tp_bad, len(bad_gt)),
        'balanced_accuracy_classified': (round((rec_bad + rec_good) / 2, 4)
                                         if rec_bad is not None and rec_good is not None else None),
        'confusion_classified': {'good_as_good': tn, 'good_as_bad': fp_bad,
                                 'bad_as_good': fn_bad, 'bad_as_bad': tp_bad},
        'pre_context_lt16': sum(1 for p in pre if p < 16), 'pre_context_negative': sum(1 for p in pre if p < 0),
        'pre_context_median': statistics.median(pre) if pre else None,
        'window_full_gt_coverage': sum(1 for r in m if r['window_gt_coverage'] >= 1.0),
        # The old key is kept for consumers of the 0006 JSON schema.
        'matched_before_first_confirm': sum(1 for r in m if r.get('before_session_confirm') is True),
        'matched_before_session_confirm': sum(1 for r in m if r.get('before_session_confirm') is True),
        'matched_with_known_session_confirm': sum(1 for r in m if r.get('session_confirm_frame') is not None),
    }


# ----------------------------------------------------------------------- main
def load_models(models_dir, videos, in_sample_model):
    import joblib
    out = {}
    for v in videos:
        if in_sample_model:
            out[v] = joblib.load(in_sample_model)
            continue
        art = joblib.load(Path(models_dir) / f'{v}.joblib')   # self-trained artifacts only
        if art.get('held_out_video') != v:
            raise ValueError(f'{v}: model held_out_video={art.get("held_out_video")!r}')
        out[v] = art
    return out


def git_commit(repo):
    try:
        return subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def run(args, classifier=None, models=None):
    caches = {p.name[:-len('.kp.bin')]: p for p in sorted(Path(args.cache_dir).glob('*.kp.bin'))}
    videos = args.video or sorted(caches)
    missing = [v for v in videos if v not in caches]
    if missing:
        raise FileNotFoundError(f'no cache for {missing} in {args.cache_dir}')
    if classifier is None and not args.count_only:
        from squat_classification import classify
        classifier = classify
    if models is None:
        models = ({v: None for v in videos} if args.count_only
                  else load_models(args.models_dir, videos, args.in_sample_model))
    conditions = [tuple(c.split('/')) for c in args.condition] if args.condition else CONDITIONS
    # absolute: AutoDetectPipeline resolves relative paths against the project root
    out = (Path(args.out_dir) / (args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    all_rows, all_extras, per_video, run_info = [], [], defaultdict(dict), {}
    for v in videos:
        data, meta = load_cache(caches[v])
        if args.require_complete and not meta.get('complete', False):
            raise ValueError(f'{v}: cache is marked incomplete')
        fps = float(meta.get('assumed_fps') or meta.get('video', {}).get('fps') or 30.0)
        mask = args.mask_conf if meta.get('source') == 'pc_ultralytics_cache' else args.board_mask_conf
        gt = load_gt(args.segmentation, v)
        if models.get(v) is not None and gt and 'held_out_person_id' in models[v] \
                and models[v]['held_out_person_id'] != gt[0]['person_id']:
            raise ValueError(f'{v}: held-out person mismatch')
        lo = args.start_frame
        hi = args.end_frame if args.end_frame is not None else int(data['frame_id'][-1])
        run_info[v] = {'cache': str(caches[v]), 'frames': int(len(data)), 'fps': fps, 'mask_conf': mask,
                       'cache_complete': meta.get('complete'), 'video_sha256': meta.get('video', {}).get('sha256'),
                       'range': [lo, hi], 'exact_15_frame_hip_gaps': exact_gap_runs(
                           [mask_conf(k, mask) for k in data['kp']])}
        for policy, gating in conditions:
            name = f'{policy}/{gating}'
            cdir = out / f'{policy}__gating_{gating}'
            cdir.mkdir(exist_ok=True)
            factory = make_factory(models[v], fps, policy, classifier, bottom_depth=args.bottom_depth,
                                   bottom_knee=args.bottom_knee, pre=args.pre_frames, post=args.post_frames)
            reps, events, stats = replay(data, fps, factory, gating, cdir / f'{v}_auto_events.jsonl', mask)
            with (cdir / f'{v}_reps.jsonl').open('w', encoding='utf-8') as f:
                for r in reps:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            squat_starts = [e for e in events if e['type'] == 'session_start' and e['exercise'] == 'squat']
            confirms = [e['frame_id'] for e in squat_starts]
            # Only gated mode can attribute a replayed rep to a squat session.
            confirm_by_session = ({e['session_id']: e['frame_id'] for e in squat_starts
                                   if e.get('session_id') is not None}
                                  if gating == 'on' else None)
            rows, extras, excluded = evaluate_video(
                v, gt, reps, lo=lo, hi=hi, min_iou=args.min_iou, condition=name,
                session_confirms=confirm_by_session)
            all_rows += rows
            all_extras += extras
            per_video[name][v] = {**summarize(rows, extras), 'gt_excluded_by_range': excluded,
                                  'squat_sessions': len(confirms), 'first_squat_confirm_frame':
                                  confirms[0] if confirms else None, **stats}
            print(f'{v} {name}: ' + json.dumps({k: per_video[name][v][k] for k in (
                'gt_reps', 'detected_matched', 'extra_detections', 'classification_failed',
                'accuracy_classified', 'accuracy_end_to_end', 'pre_context_negative')}), flush=True)
    overall = {}
    for policy, gating in conditions:
        name = f'{policy}/{gating}'
        overall[name] = summarize([r for r in all_rows if r['condition'] == name],
                                  [e for e in all_extras if e['condition'] == name])
    fields = sorted({k for r in all_rows for k in r})
    with (out / 'reps_eval.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    with (out / 'extra_detections.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['condition', 'video_id', 'extra', 'det_start', 'det_end', 'posture'])
        w.writeheader()
        w.writerows(all_extras)
    summary = {'overall': overall, 'per_video': per_video, 'inputs': run_info,
               'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               'model_kind': 'IN-SAMPLE (all nine people)' if args.in_sample_model else
                             ('count_only' if args.count_only else 'person-held-out'),
               'code_commit': git_commit(Path(__file__).resolve().parents[1]),
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'definitions': __doc__}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                      encoding='utf-8')
    lines = ['| condition | GT | matched | extra | failed | acc(classified) | acc(end-to-end) | '
             'Bad recall(cls) | Bad recall(e2e) | pre<16 | pre<0 |', '|' + '---|' * 11]
    for name, s in overall.items():
        lines.append(f"| {name} | {s['gt_reps']} | {s['detected_matched']} | {s['extra_detections']} | "
                     f"{s['classification_failed']} | {s['accuracy_classified']} | {s['accuracy_end_to_end']} | "
                     f"{s['bad_recall_classified']} | {s['bad_recall_end_to_end']} | "
                     f"{s['pre_context_lt16']} | {s['pre_context_negative']} |")
    (out / 'summary.md').write_text(f"# Offline stream evaluation ({summary['model_kind']})\n\n"
                                    + '\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'Written: {out}')
    return out, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', type=Path, default=Path('data/cache/kp_full/rehab24_camera17_yolov8m'))
    ap.add_argument('--segmentation', type=Path, default=Path('data/raw/REHAB24-6/Segmentation.csv'))
    ap.add_argument('--models-dir', type=Path, default=Path('models/lopo_heldout/PlusAsymmetry180_crop10'))
    ap.add_argument('--in-sample-model', type=Path, help='e.g. the deployed all-nine model (results flagged IN-SAMPLE)')
    ap.add_argument('--count-only', action='store_true')
    ap.add_argument('--video', action='append')
    ap.add_argument('--condition', action='append', help='policy/gating, e.g. overlap/off (repeatable)')
    ap.add_argument('--out-dir', type=Path, default=Path('results/offline_stream_eval'))
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--end-frame', type=int, default=None, help='default: last cached frame')
    ap.add_argument('--min-iou', type=float, default=0.2)
    ap.add_argument('--mask-conf', type=float, default=0.30, help='applied to raw PC caches')
    ap.add_argument('--board-mask-conf', type=float, default=0.0, help='board dumps are already adapter output')
    ap.add_argument('--bottom-depth', type=float, default=0.20)
    ap.add_argument('--bottom-knee', type=float, default=120.0)
    ap.add_argument('--pre-frames', type=int, default=16)
    ap.add_argument('--post-frames', type=int, default=22)
    ap.add_argument('--run-name', default=None, help='output subfolder (default: timestamp)')
    ap.add_argument('--require-complete', action=argparse.BooleanOptionalAction, default=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
