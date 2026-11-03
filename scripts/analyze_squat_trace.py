"""Summarize per-frame CSV emitted by run_squat_hailo_pose.py.

Only standard-library modules; safe in Hailo's existing virtual environment.
The trace captures the already running counter's state and does not change it.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def number(row, key):
    value = row.get(key, '')
    if value is None or value == '':
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def contiguous_runs(rows, predicate):
    """Return inclusive callback-frame intervals for contiguous matching rows."""
    spans = []
    first = last = None
    for row in rows:
        frame = int(row['frame_id'])
        if predicate(row):
            if first is None:
                first = frame
            elif frame != last + 1:
                spans.append((first, last))
                first = frame
            last = frame
        elif first is not None:
            spans.append((first, last))
            first = last = None
    if first is not None:
        spans.append((first, last))
    return spans


def describe_candidates(rows):
    """Check whether multi-detections represent stable tracked people or duplicates."""
    if not rows or 'person_candidates_json' not in rows[0]:
        print('Person candidate details unavailable; update the runner and rerun.')
        return

    print('\n=== Person candidate tracker details ===')
    windows = [(210, 240), (241, 267), (268, 307), (308, 330)]
    for lo, hi in windows:
        counts = Counter()
        unknown = 0
        areas = {}
        centers = {}
        for row in rows:
            frame = int(row['frame_id'])
            if not lo <= frame <= hi:
                continue
            for det in json.loads(row['person_candidates_json'] or '[]'):
                key = str(det.get('track_id')) if det.get('track_id') is not None else '(no ID)'
                counts[key] += 1
                bbox = det.get('bbox', [0, 0, 0, 0])
                centers.setdefault(key, []).append(bbox[0] + bbox[2] / 2)
                areas.setdefault(key, []).append(bbox[2] * bbox[3])
                unknown += not det.get('has_landmarks', False)
        print(f'  frames={lo}-{hi}: ' +
              (', '.join(
                  f'id={key}: {count} detections, '
                  f'mean_center_x={sum(centers[key])/len(centers[key]):.2f}, '
                  f'mean_bbox_area={sum(areas[key])/len(areas[key]):.3f}'
                  for key, count in counts.most_common(6)
              ) or 'no people') +
              f'; detections_without_landmarks={unknown}')

    for frame in (240, 250, 260, 268, 280, 300, 308, 317, 325):
        row = next((r for r in rows if int(r['frame_id']) == frame), None)
        if row is not None:
            cand = json.loads(row['person_candidates_json'] or '[]')
            print(f'  frame={frame} candidates={json.dumps(cand, ensure_ascii=False)}')


def describe(rows, start, end):
    selected = [r for r in rows if start <= int(r['frame_id']) <= end]
    print(f'\n=== Frames {start}..{end} ({len(selected)} callback rows) ===')
    if not selected:
        return
    people = Counter('multi' if int(r['person_count']) > 1 else
                     'one' if int(r['person_count']) == 1 else 'zero'
                     for r in selected)
    valid = sum(int(r['valid_pose']) for r in selected)
    print(f'People counts: zero={people["zero"]} one={people["one"]} '
          f'multi={people["multi"]}; valid_pose={valid}/{len(selected)}')
    print('States:', dict(Counter(r['state_after'] for r in selected)))

    gaps = contiguous_runs(selected, lambda r: int(r['valid_pose']) == 0)
    gaps.sort(key=lambda x: x[1] - x[0], reverse=True)
    print('Longest missing-pose runs:',
          ', '.join(f'{a}-{b} ({b-a+1}f)' for a, b in gaps[:8]) or 'none')
    multi = contiguous_runs(selected, lambda r: int(r['person_count']) > 1)
    multi.sort(key=lambda x: x[1] - x[0], reverse=True)
    print('Longest multi-person runs:',
          ', '.join(f'{a}-{b} ({b-a+1}f)' for a, b in multi[:8]) or 'none')

    depths = [number(r, 'relative_hip_depth') for r in selected]
    knees = [number(r, 'smoothed_knee_angle') for r in selected]
    depths = [x for x in depths if math.isfinite(x)]
    knees = [x for x in knees if math.isfinite(x)]
    print('Smoothed depth max:', f'{max(depths):.3f}' if depths else 'not available')
    print('Smoothed knee min:', f'{min(knees):.1f}' if knees else 'not available')
    print('Counter increments:',
          sum(int(r['counter_after']) > int(r['counter_before']) for r in selected))

    print('State changes / calibration reset:')
    found = False
    for row in selected:
        reset = row['standing_hip_before'] and not row['standing_hip_after']
        transition = row['state_before'] != row['state_after']
        inc = int(row['counter_after']) > int(row['counter_before'])
        if reset or transition or inc:
            found = True
            print(f'  frame={row["frame_id"]} people={row["person_count"]} '
                  f'valid={row["valid_pose"]} '
                  f'{row["state_before"]}->{row["state_after"]} '
                  f'missing={row["missing_before"]}->{row["missing_after"]} '
                  f'depth={row["relative_hip_depth"] or "-"} '
                  f'knee={row["smoothed_knee_angle"] or "-"}'
                  + (' [STANDING CALIBRATION CLEARED]' if reset else '')
                  + (' [REP COUNTED]' if inc else ''))
    if not found:
        print('  none')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path, help='SQUAT_TRACE_OUTPUT CSV')
    args = parser.parse_args()
    with args.path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit('No frame rows; check trace range and file playback')
    frame_ids = [int(r['frame_id']) for r in rows]
    print(f'Trace frames: {frame_ids[0]}..{frame_ids[-1]} ({len(rows)} rows)')
    describe(rows, 210, 330)
    describe(rows, 340, 460)
    describe_candidates(rows)
    describe(rows, frame_ids[0], frame_ids[-1])
    print('\nInterpretation: missing runs >=15 frames can clear the counter '
          'calibration at 30 FPS. A BOTTOM state may be triggered by '
          'depth >=0.20 OR knee <=120. Inspect whether the subject had '
          'a valid pose and reached BOTTOM and ASCENDING before changing thresholds.')


if __name__ == '__main__':
    main()
