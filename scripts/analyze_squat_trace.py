"""Summarize per-frame CSV emitted by run_squat_hailo_pose.py.

Only standard-library modules; safe in Hailo's existing virtual environment.
The trace captures the already running counter's state and does not change it.
"""
import argparse
import csv
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
    describe(rows, frame_ids[0], frame_ids[-1])
    print('\nInterpretation: missing runs >=15 frames can clear the counter '
          'calibration at 30 FPS. A BOTTOM state may be triggered by '
          'depth >=0.20 OR knee <=120. Inspect whether the subject had '
          'a valid pose and reached BOTTOM and ASCENDING before changing thresholds.')


if __name__ == '__main__':
    main()
