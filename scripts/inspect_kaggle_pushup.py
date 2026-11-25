#!/usr/bin/env python3
"""Read-only audit of Kaggle push-up clips, with contact sheets for source review."""
import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def numeric_id(path):
    m = re.search(r'(\d+)$', path.stem)
    return int(m.group(1)) if m else -1


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = None
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = 0
    width = height = 0
    # Keep only three small thumbnails, not the full video in memory.
    wanted = [round(max(declared - 1, 0) * q) for q in (0.15, 0.5, 0.85)]
    thumbs = [None, None, None]
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        for i, n in enumerate(wanted):
            if n == frames:
                thumbs[i] = thumbnail(frame)
        last = frame
        frames += 1
    cap.release()
    if last is not None:
        for i in range(3):
            if thumbs[i] is None:
                thumbs[i] = thumbnail(last)
    return fps, declared, frames, width, height, thumbs


def thumbnail(frame):
    h, w = frame.shape[:2]
    s = min(250 / w, 140 / h)
    im = cv2.resize(frame, (max(1, round(w*s)), max(1, round(h*s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((140, 250, 3), 242, np.uint8)
    y, x = (140-im.shape[0])//2, (250-im.shape[1])//2
    canvas[y:y+im.shape[0], x:x+im.shape[1]] = im
    return canvas


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def label_info(path, root):
    item = {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size}
    try:
        a = np.load(path, allow_pickle=False)
        item.update(shape=list(a.shape), dtype=str(a.dtype))
        if a.dtype.kind in 'biufUS':
            flat = a.reshape(-1)
            item['first_30_values'] = [str(x) for x in flat[:30]]
            if flat.size <= 1000000:
                values, counts = np.unique(flat, return_counts=True)
                item['unique_count'] = len(values)
                item['unique_values_first_30'] = [{'value': str(v), 'count': int(n)} for v, n in zip(values[:30], counts[:30])]
    except Exception as e:
        item['error'] = str(e)
        item['note'] = 'Object arrays are not loaded with pickle enabled.'
    return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path.home()/'capstone2/datasets/Kaggle-Pushup')
    p.add_argument('--out-dir', type=Path, default=Path('results/pushup/kaggle_audit'))
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for folder, label in [('Correct sequence', 'good'), ('Wrong sequence', 'bad')]:
        d = root / folder
        if not d.is_dir():
            raise SystemExit(f'Missing folder: {d}')
        clips.extend((path, label) for path in d.rglob('*') if path.is_file() and path.suffix.lower()=='.mp4')
    clips.sort(key=lambda x: (numeric_id(x[0]), x[1], str(x[0])))
    if not clips:
        raise SystemExit('No MP4 clips found')
    rows, sheets = [], []
    hashes = defaultdict(list)
    for i, (path, label) in enumerate(clips):
        fps, declared, frames, width, height, thumbs = inspect_video(path)
        row = dict(sample_id=f'KPU_{i+1:03d}', relative_path=str(path.relative_to(root)), filename=path.name,
                   numeric_id=numeric_id(path), label=label, fps=fps, declared_frames=declared,
                   decoded_frames=frames, width=width, height=height,
                   duration_sec=round(frames/fps, 4) if fps else None, sha256=sha256(path),
                   status='ok' if frames and fps else 'decode_or_fps_error')
        rows.append(row)
        hashes[row['sha256']].append(row)
        # Sort sheets by numerical filename ID to make likely neighboring clips easy to compare.
        if i % 10 == 0:
            canvas = np.full((10*160+34, 1040, 3), 255, np.uint8)
            cv2.putText(canvas, 'ID / original label | frames near 15%, 50%, 85%', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (25,25,25), 1, cv2.LINE_AA)
        y = 34 + (i % 10)*160
        cv2.putText(canvas, f"{row['sample_id']} / {row['numeric_id']}", (8,y+32), cv2.FONT_HERSHEY_SIMPLEX,.60,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(canvas, f"{label} | {frames} frames", (8,y+58), cv2.FONT_HERSHEY_SIMPLEX,.52,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(canvas, f"{row['duration_sec']} s", (8,y+82), cv2.FONT_HERSHEY_SIMPLEX,.52,(20,20,20),1,cv2.LINE_AA)
        for j, thumb in enumerate(thumbs):
            if thumb is not None:
                x=270+j*255
                canvas[y:y+140,x:x+250]=thumb
        if i % 10 == 9 or i == len(clips)-1:
            dest=out/f'contact_{i//10+1:02d}.jpg'
            if not cv2.imwrite(str(dest),canvas[:34+(i%10+1)*160], [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise RuntimeError(f'Could not write {dest}')
            sheets.append(dest)
        print(f"[{i+1}/{len(clips)}] {row['sample_id']} {label} {frames} frames {row['duration_sec']}s",flush=True)
    dup=[]
    for h, group in hashes.items():
        if len(group)>1:
            dup.append({'sha256':h,'label_conflict':len({r['label'] for r in group})>1,'members':[r['relative_path'] for r in group]})
    labels=[label_info(path,root) for path in sorted((root/'labels').rglob('*.npy'))]
    durations=[r['duration_sec'] for r in rows if r['duration_sec'] is not None]
    summary=dict(root=str(root),clip_count=len(rows),labels=dict(Counter(r['label'] for r in rows)),
                 fps_counts=dict(Counter(str(r['fps']) for r in rows)),resolution_counts=dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
                 duration_sec=dict(min=min(durations),median=float(np.median(durations)),max=max(durations),total=sum(durations)) if durations else None,
                 decode_errors=[r['sample_id'] for r in rows if r['status']!='ok'],
                 frame_count_mismatches=[r['sample_id'] for r in rows if r['declared_frames']!=r['decoded_frames']],
                 exact_duplicate_groups=len(dup),conflicting_duplicate_groups=sum(d['label_conflict'] for d in dup),npy_files=labels,
                 note='Contact sheets are for visual source/session grouping only; filenames and exact hashes do not establish independent subjects. No label-to-clip mapping inferred from NPY arrays.')
    write_csv(out/'inventory.csv',rows,list(rows[0]))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'duplicates.json').write_text(json.dumps(dup,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(out/'kaggle_audit.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in [out/'inventory.csv',out/'summary.json',out/'duplicates.json',*sheets]:
            z.write(f,arcname=f.name)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Saved:',out/'kaggle_audit.zip')

if __name__=='__main__':
    main()
