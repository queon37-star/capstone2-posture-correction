"""Post-hoc evaluation of streaming squat predictions on *whole* REHAB24-6 videos.

Important: ground truth is read for evaluation only, never for event detection.
This evaluates repetition detection, interval quality, and classification separately.
The prototype classifier is trained on these subjects; scores are *not* an
unseen-subject estimate of generalization performance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def segment_iou(a, b):
    """Inclusive interval IoU; zero for nonoverlapping ranges."""
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0


def match_segments(gt, pred, min_iou=0.2):
    """One-to-one greedy matching by descending overlap."""
    edges = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            score = segment_iou((g['start'], g['end']), (p['start'], p['end']))
            if score >= min_iou:
                edges.append((score, i, j))
    matches = []
    matched_g, matched_p = set(), set()
    for score, i, j in sorted(edges, key=lambda x: (-x[0], x[1], x[2])):
        if i not in matched_g and j not in matched_p:
            matched_g.add(i)
            matched_p.add(j)
            matches.append((i, j, score))
    return matches, matched_g, matched_p


def evaluate(gt, pred, min_iou=0.2):
    matches, used_gt, used_pred = match_segments(gt, pred, min_iou)
    rows = []
    for gi, pj, overlap in matches:
        g, p = gt[gi], pred[pj]
        outcome = 'correct' if p['posture'] == ('good' if g['label'] == 1 else 'bad') else 'wrong'
        if p['posture'] not in ('good', 'bad'):
            outcome = 'unknown'
        rows.append({
            'status': 'matched', 'gt_rep': g['rep'], 'gt_start': g['start'], 'gt_end': g['end'],
            'gt_label': g['label'], 'pred_count': p.get('count'),
            'pred_start': p['start'], 'pred_end': p['end'],
            'pred_posture': p['posture'], 'interval_iou': round(overlap, 4),
            'start_error_frames': p['start'] - g['start'],
            'end_error_frames': p['end'] - g['end'], 'classification': outcome,
        })
    for gi, g in enumerate(gt):
        if gi not in used_gt:
            rows.append({'status': 'missed', 'gt_rep': g['rep'], 'gt_start': g['start'],
                         'gt_end': g['end'], 'gt_label': g['label']})
    for pj, p in enumerate(pred):
        if pj not in used_pred:
            rows.append({'status': 'extra', 'pred_count': p.get('count'),
                         'pred_start': p['start'], 'pred_end': p['end'],
                         'pred_posture': p['posture']})
    rows.sort(key=lambda x: (x.get('gt_start', x.get('pred_start', 0)), x['status']))
    matched = [r for r in rows if r['status'] == 'matched']
    correct = [r for r in matched if r['classification'] == 'correct']
    bad_gt = sum(g['label'] == 0 for g in gt)
    good_gt = sum(g['label'] == 1 for g in gt)
    bad_match = [r for r in matched if r['gt_label'] == 0]
    good_match = [r for r in matched if r['gt_label'] == 1]
    bad_hit = sum(r['pred_posture'] == 'bad' for r in bad_match)
    good_hit = sum(r['pred_posture'] == 'good' for r in good_match)
    bad_pred_match = sum(r['pred_posture'] == 'bad' for r in matched)
    extra_bad = sum(pred[i]['posture'] == 'bad' for i in range(len(pred)) if i not in used_pred)
    summary = {
        'gt_repetitions': len(gt), 'predicted_repetitions': len(pred),
        'matched': len(matched), 'missed': len(gt) - len(used_gt),
        'extra': len(pred) - len(used_pred),
        'rep_detection_recall': len(matched) / len(gt) if gt else None,
        'rep_detection_precision': len(matched) / len(pred) if pred else None,
        'avg_interval_iou': mean(r['interval_iou'] for r in matched) if matched else None,
        'start_mae_frames': mean(abs(r['start_error_frames']) for r in matched) if matched else None,
        'end_mae_frames': mean(abs(r['end_error_frames']) for r in matched) if matched else None,
        'start_bias_frames': mean(r['start_error_frames'] for r in matched) if matched else None,
        'end_bias_frames': mean(r['end_error_frames'] for r in matched) if matched else None,
        'matched_classifier_accuracy': len(correct) / len(matched) if matched else None,
        'matched_correct_predictions': len(correct),
        'matched_unknown_predictions': sum(r['classification'] == 'unknown' for r in matched),
        'gt_correct': good_gt, 'gt_incorrect': bad_gt,
        'matched_correct_reps': len(good_match), 'matched_incorrect_reps': len(bad_match),
        'detected_and_classified_correct': good_hit,
        'detected_and_classified_incorrect': bad_hit,
        'incorrect_recall_matched': bad_hit / len(bad_match) if bad_match else None,
        'incorrect_end_to_end_detection_rate': bad_hit / bad_gt if bad_gt else None,
        'correct_recall_matched': good_hit / len(good_match) if good_match else None,
        'matched_pred_bad': bad_pred_match,
        'extra_pred_bad': extra_bad,
        'incorrect_precision_matched': bad_hit / bad_pred_match if bad_pred_match else None,
    }
    return summary, rows


def load_gt(path, video, start, end):
    with path.open(newline='', encoding='utf-8-sig') as f:
        allrows = list(csv.DictReader(f, delimiter=';'))
    results = []
    for r in allrows:
        if r['video_id'] != video or int(r['exercise_id']) != 6:
            continue
        a, b = int(r['first_frame']), int(r['last_frame'])
        if start <= a <= b <= end:
            results.append({'start': a, 'end': b, 'rep': int(r['repetition_number']),
                            'label': int(r['correctness'])})
    return sorted(results, key=lambda r: r['start'])


def load_pred(path, start, end):
    results = []
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('type') != 'rep':
                continue
            a, b = int(r['start_frame']), int(r['end_frame'])
            if not a <= b:
                raise ValueError(f'Invalid predicted interval on line {i}')
            if start <= a <= b <= end:
                results.append({'start': a, 'end': b, 'posture': r.get('posture', 'unknown'),
                                'count': r.get('count')})
    return sorted(results, key=lambda r: r['start'])


def video_final_frame(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        raise ValueError('Video frame count unavailable; provide --end-frame')
    return count - 1


def fmt(value, digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True)
    ap.add_argument('--events', type=Path, default=None)
    ap.add_argument('--annotations', type=Path, default=Path('data/raw/REHAB24-6/Segmentation.csv'))
    ap.add_argument('--video-file', type=Path, default=None)
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--end-frame', type=int, default=None)
    ap.add_argument('--min-iou', type=float, default=0.2)
    ap.add_argument('--out-dir', type=Path, default=Path('results/stream_test'))
    args = ap.parse_args()
    if not 0 < args.min_iou <= 1:
        ap.error('--min-iou must be in (0,1]')
    events = args.events or (Path('results/stream_test') / f'{args.video}_events.jsonl')
    videopath = args.video_file or (Path('data/raw/REHAB24-6') / f'{args.video}-Camera17-30fps.mp4')
    end = args.end_frame if args.end_frame is not None else video_final_frame(videopath)
    if end <= args.start_frame:
        ap.error('--end-frame must exceed --start-frame')
    gt = load_gt(args.annotations, args.video, args.start_frame, end)
    pred = load_pred(events, args.start_frame, end)
    summary, rows = evaluate(gt, pred, args.min_iou)
    summary.update({'video': args.video, 'start_frame': args.start_frame,
                    'end_frame': end, 'matching_iou_threshold': args.min_iou})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonpath = args.out_dir / f'{args.video}_evaluation.json'
    csvpath = args.out_dir / f'{args.video}_matches.csv'
    jsonpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    keys = ['status','gt_rep','gt_start','gt_end','gt_label','pred_count','pred_start','pred_end',
            'pred_posture','interval_iou','start_error_frames','end_error_frames','classification']
    with csvpath.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Video {args.video} frames {args.start_frame}..{end} (no GT used during stream inference)')
    print(f"Repetitions GT={summary['gt_repetitions']} predicted={summary['predicted_repetitions']} "
          f"matched={summary['matched']} missed={summary['missed']} extra={summary['extra']}")
    print(f"Boundary: IoU={fmt(summary['avg_interval_iou'])} "
          f"start_MAE={fmt(summary['start_mae_frames'], 2)} end_MAE={fmt(summary['end_mae_frames'], 2)} "
          f"start_bias={fmt(summary['start_bias_frames'], 2)} end_bias={fmt(summary['end_bias_frames'], 2)} frames")
    print(f"Matched classifier accuracy: {summary['matched_correct_predictions']}/{summary['matched']} "
          f"({fmt(summary['matched_classifier_accuracy'])})")
    print(f"Incorrect GT={summary['gt_incorrect']}, matched_bad={summary['matched_incorrect_reps']}, "
          f"predicted_bad_correctly={summary['detected_and_classified_incorrect']}, "
          f"matched_bad_recall={fmt(summary['incorrect_recall_matched'])}, "
          f"end_to_end_bad_detection={fmt(summary['incorrect_end_to_end_detection_rate'])}")
    print(f"Correct GT={summary['gt_correct']}, matched_good={summary['matched_correct_reps']}, "
          f"predicted_good_correctly={summary['detected_and_classified_correct']}")
    print('Match details (gt_rep / true / pred / interval / IoU):')
    for row in rows:
        if row['status']=='matched':
            print(f"  #{row['gt_rep']:02d}: {'incorrect' if row['gt_label']==0 else 'correct':9s} "
                  f"-> {row['pred_posture']:7s} "
                  f"GT={row['gt_start']}-{row['gt_end']} "
                  f"pred={row['pred_start']}-{row['pred_end']} "
                  f"IoU={row['interval_iou']:.2f}")
        elif row['status']=='missed':
            print(f"  #{row['gt_rep']:02d}: MISSED GT={row['gt_start']}-{row['gt_end']}")
        else:
            print(f"  EXTRA #{row.get('pred_count')} pred={row['pred_start']}-{row['pred_end']}")
    print('Saved:', jsonpath, csvpath)
    print('Caution: this model is refitted on these nine subjects; accuracy is not unseen-person LOPO.')


if __name__=='__main__':
    main()
