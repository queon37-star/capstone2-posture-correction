"""Inventory of the local REHAB24-6 copy (track B, step 0). Reads only; changes nothing.

Answers: which exercises/persons/cameras/labels exist locally, how many
repetitions each has, which videos are actually on disk, and which persons the
current squat model used. Output: console summary + CSV/Markdown files.

  python scripts/inventory_rehab24.py                 # data/raw/REHAB24-6
  python scripts/inventory_rehab24.py --raw <dir> --out results/dataset_inventory

Exercise names follow the dataset description (Zenodo record 13305826):
Ex3 push-ups are performed with the hands on a TABLE (not floor push-ups).
"""
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EXERCISES = {'1': 'arm_abduction', '2': 'arm_VW', '3': 'pushup_table',
             '4': 'leg_abduction', '5': 'lunge', '6': 'squat'}


def read_segmentation(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = ';' if sample.count(';') >= sample.count(',') else ','
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', default='data/raw/REHAB24-6')
    ap.add_argument('--processed-index', default='data/processed/REHAB24-6/camera17_yolov8m_tracked/index.csv')
    ap.add_argument('--out', default='results/dataset_inventory')
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_segmentation(raw / 'Segmentation.csv')
    columns = list(rows[0]) if rows else []
    orientation_cols = [c for c in columns if 'orientation' in c.lower()]
    extra_cols = [c for c in columns if c not in {
        'video_id', 'repetition_number', 'exercise_id', 'person_id',
        'first_frame', 'last_frame', 'correctness', *orientation_cols}]
    lines = [f'# REHAB24-6 local inventory', '', f'- Segmentation.csv rows: {len(rows)}',
             f'- columns: {", ".join(columns)}',
             f'- persons in CSV: {sorted({r["person_id"] for r in rows}, key=lambda x: int(x))}', '']

    # 1) per exercise
    by_ex = defaultdict(list)
    for r in rows:
        by_ex[r['exercise_id']].append(r)
    ex_rows = []
    for ex in sorted(by_ex, key=int):
        rs = by_ex[ex]
        frames = [int(r['last_frame']) - int(r['first_frame']) + 1 for r in rs]
        corr = Counter(r['correctness'] for r in rs)
        ex_rows.append([ex, EXERCISES.get(ex, '?'), len(rs), len({r['person_id'] for r in rs}),
                        len({r['video_id'] for r in rs}), corr.get('1', 0), corr.get('0', 0),
                        round(statistics.median(frames)), min(frames), max(frames)])
    header = ['exercise_id', 'name', 'reps', 'persons', 'videos', 'correct(1)', 'incorrect(0)',
              'median_frames', 'min_frames', 'max_frames']
    write_csv(out / 'per_exercise.csv', header, ex_rows)
    lines += ['## Per exercise', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(map(str, r)) + ' |' for r in ex_rows] + ['']

    # 2) exercise x person x label
    cell = Counter((r['exercise_id'], r['person_id'], r['correctness']) for r in rows)
    persons = sorted({r['person_id'] for r in rows}, key=int)
    ep_rows = [[ex, EXERCISES.get(ex, '?'), p, cell[(ex, p, '1')], cell[(ex, p, '0')]]
               for ex in sorted(by_ex, key=int) for p in persons]
    write_csv(out / 'per_exercise_person.csv',
              ['exercise_id', 'name', 'person_id', 'correct', 'incorrect'], ep_rows)
    lines += ['## Exercise x person (correct/incorrect)', '',
              '| ex | ' + ' | '.join(f'P{p}' for p in persons) + ' |', '|' + '---|' * (len(persons) + 1)]
    for ex in sorted(by_ex, key=int):
        lines.append(f'| {ex} {EXERCISES.get(ex, "?")} | ' + ' | '.join(
            f'{cell[(ex, p, "1")]}/{cell[(ex, p, "0")]}' for p in persons) + ' |')
    lines.append('')

    # 3) orientation / extra label columns
    for col in orientation_cols + extra_cols:
        dist = defaultdict(Counter)
        for r in rows:
            dist[r['exercise_id']][r.get(col, '')] += 1
        lines += [f'## {col} by exercise', '']
        lines += [f'- Ex{ex} {EXERCISES.get(ex, "?")}: {dict(sorted(dist[ex].items()))}'
                  for ex in sorted(dist, key=int)] + ['']

    # 4) video files on disk
    videos = sorted({r['video_id'] for r in rows})
    ex_of_video = defaultdict(set)
    for r in rows:
        ex_of_video[r['video_id']].add(r['exercise_id'])
    cams = Counter()
    vid_rows = []
    for v in videos:
        files = sorted(p.name for p in raw.glob(f'{v}-Camera*'))
        cam_ids = sorted({n.split('-Camera')[1].split('-')[0] for n in files})
        cams.update(cam_ids)
        vid_rows.append([v, ';'.join(sorted(ex_of_video[v], key=int)),
                         ';'.join(cam_ids) or 'MISSING', ';'.join(files)])
    write_csv(out / 'videos.csv', ['video_id', 'exercise_ids', 'cameras_on_disk', 'files'], vid_rows)
    missing = [r[0] for r in vid_rows if r[2] == 'MISSING']
    lines += ['## Videos on disk', '', f'- videos referenced: {len(videos)}, '
              f'with at least one camera file: {len(videos) - len(missing)}',
              f'- camera counts: {dict(sorted(cams.items()))}',
              f'- missing: {missing[:20]}{" ..." if len(missing) > 20 else ""}', '']

    # 5) other files (skeletons etc.) by top-level folder and extension
    tree = Counter()
    for p in raw.rglob('*'):
        if p.is_file():
            top = p.relative_to(raw).parts[0] if len(p.relative_to(raw).parts) > 1 else '.'
            tree[(top, p.suffix.lower())] += 1
    lines += ['## Files under raw/ (folder, extension: count)', '']
    lines += [f'- {top} {ext or "(none)"}: {n}' for (top, ext), n in sorted(tree.items())] + ['']

    # 6) persons used by the current squat model
    idx = Path(args.processed_index)
    if idx.is_file():
        with idx.open(newline='', encoding='utf-8') as f:
            used = list(csv.DictReader(f))
        used_p = sorted({u['person_id'] for u in used if u.get('person_id')}, key=int)
        all_p = sorted({r['person_id'] for r in by_ex.get('6', [])}, key=int)
        lines += ['## Current squat model data (processed index)', '',
                  f'- reps in index: {len(used)}; persons used: {used_p}',
                  f'- Ex6 persons in Segmentation.csv: {all_p}; not used: '
                  f'{sorted(set(all_p) - set(used_p), key=int)}', '']
    (out / 'INVENTORY.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'columns.json').write_text(json.dumps(columns, ensure_ascii=False), encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nWritten: {out}/INVENTORY.md, per_exercise.csv, per_exercise_person.csv, videos.csv')


if __name__ == '__main__':
    main()
