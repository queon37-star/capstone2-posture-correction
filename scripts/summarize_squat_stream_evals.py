"""Aggregate nine per-video validate_squat_stream.py JSON summaries.

Run with --eval-dir results/stream_test_bd020_eval (in-sample) or
--eval-dir results/stream_lopo_eval (held-out classifier weights).
Never substitutes one denominator for another: reports detected-rep
classification and end-to-end incorrect coverage separately.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

VIDEOS = ['PM_008','PM_022','PM_029','PM_038','PM_043',
          'PM_105','PM_113','PM_118','PM_126']


def aggregate(evaluations):
    sumfield = lambda name: sum(int(r[name]) for r in evaluations)
    n_gt = sumfield('gt_repetitions')
    n_pred = sumfield('predicted_repetitions')
    n_match = sumfield('matched')
    n_bad = sumfield('gt_incorrect')
    n_good = sumfield('gt_correct')
    bad_correct = sumfield('detected_and_classified_incorrect')
    good_correct = sumfield('detected_and_classified_correct')
    n_true = bad_correct + good_correct
    n_pred_bad = sumfield('matched_pred_bad') + sumfield('extra_pred_bad')
    w_iou = sum((r['avg_interval_iou'] or 0) * int(r['matched']) for r in evaluations)
    return {
        'gt_repetitions': n_gt,
        'predicted_repetitions': n_pred,
        'matched': n_match,
        'missed': sumfield('missed'),
        'extra': sumfield('extra'),
        'detection_recall': n_match/n_gt if n_gt else None,
        'detection_precision': n_match/n_pred if n_pred else None,
        'matched_classifier_accuracy': n_true/n_match if n_match else None,
        'matched_correct': n_true,
        'ground_truth_correct': n_good,
        'ground_truth_incorrect': n_bad,
        'detected_and_classified_correct': good_correct,
        'detected_and_classified_incorrect': bad_correct,
        'incorrect_end_to_end_recall': bad_correct/n_bad if n_bad else None,
        'bad_precision_including_extra_events': bad_correct/n_pred_bad if n_pred_bad else None,
        'mean_interval_iou_over_matches': w_iou/n_match if n_match else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eval-dir', type=Path, required=True)
    ap.add_argument('--save', type=Path, default=None)
    args = ap.parse_args()
    missing = [vid for vid in VIDEOS if not (args.eval_dir/f'{vid}_evaluation.json').is_file()]
    if missing:
        raise FileNotFoundError(f'Missing evaluation files for: {", ".join(missing)}')
    records = []
    for vid in VIDEOS:
        src = args.eval_dir/f'{vid}_evaluation.json'
        row = json.loads(src.read_text(encoding='utf-8'))
        if row['video'] != vid:
            raise ValueError(f'Mismatched video in {src}: {row.get("video")}')
        records.append(row)
        print(f'{vid}: reps={row["matched"]}/{row["gt_repetitions"]} '
              f'good+bad correct={row["matched_correct_predictions"]}/{row["matched"]} '
              f'bad={row["detected_and_classified_incorrect"]}/{row["gt_incorrect"]}')
    overall = aggregate(records)
    print('\nAGGREGATE')
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(overall, indent=2), encoding='utf-8')
        print('Saved:', args.save)


if __name__ == '__main__':
    main()
