#!/usr/bin/env python3
"""REHAB24-6 inventory, human-reviewed anchors, cache extraction and QA.
Run from pose_training_offline with this file in scripts/. Existing caches are read-only.
"""
import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / 'capstone2/pose_training/REHAB24-6'
WEIGHTS = Path.home() / 'capstone2/pose_training/yolov8m-pose.pt'
WORK = Path('data/preprocess/rehab24')
LEGACY = Path('data/cache/kp_full/rehab24_camera17_yolov8m')
NEW = Path('data/cache/kp_full/rehab24_all_yolov8m')
NAMES = {1:'arm_abduction', 2:'arm_vw', 3:'table_pushup', 4:'leg_abduction', 5:'lunge', 6:'squat'}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cache_paths(v):
    p = Path(v['cache_dir']) / v['video_id']
    return [Path(str(p) + s) for s in ('.kp.bin', '.kp.bin.json', '.frames.npz')]


def check_cache(v, model_hash, verify_video=False, verify_anchor=None):
    from extract_video_kp_cache import KP_DUMP_DTYPE, sha256_file
    files = cache_paths(v)
    if not any(p.exists() for p in files):
        return 'missing'
    if not all(p.is_file() for p in files):
        return 'partial'
    try:
        m = load_json(files[1]); n = int(m['frames_decoded'])
        if (not m['complete'] or n != v['frames'] or m['video']['video_id'] != v['video_id']
                or abs(float(m['video']['fps']) - v['fps']) > .001
                or m['video']['size_bytes'] != Path(v['path']).stat().st_size
                or m['format'] != 'capstone2-kp-cache' or m['version'] != 1
                or m['pose_model']['weights_sha256'] != model_hash
                or m['pose_model']['imgsz'] != 640 or m['pose_model']['conf'] != .25
                or files[0].stat().st_size != n * KP_DUMP_DTYPE.itemsize):
            return 'incompatible'
        if verify_video and m['video']['sha256'] != sha256_file(v['path']):
            return 'video_hash_mismatch'
        if verify_anchor is not None and not np.allclose(m['target_selection']['anchor_bbox'], verify_anchor['bbox'], rtol=0, atol=.001):
            return 'anchor_mismatch'
        with np.load(files[2], allow_pickle=False) as z:
            if len(z['status']) != n or not np.array_equal(z['frame_id'], np.arange(n)):
                return 'frame_index_mismatch'
        return 'complete'
    except Exception as e:
        return 'invalid:' + str(e)


def build_manifest(root, weights):
    from extract_video_kp_cache import sha256_file
    groups = defaultdict(list)
    with (root / 'Segmentation.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            ex = int(r['exercise_id']); groups[(ex, r['video_id'])].append(r)
    model_hash = sha256_file(weights)
    rows = []; seen = set()
    for exdir in sorted((root / 'videos').glob('Ex*')):
        ex = int(exdir.name[2:])
        for p in sorted(exdir.glob('*.mp4')):
            m = re.fullmatch(r'(.+)-Camera(17|18)-30fps(?:-transposed)?\.mp4', p.name)
            if not m:
                raise ValueError(f'Unexpected file name: {p}')
            vid, cam = m[1], int(m[2]); key = (ex, vid, cam)
            if key in seen or (ex,vid) not in groups:
                raise ValueError(f'Duplicate or no annotation: {key}')
            seen.add(key)
            gt = sorted(groups[(ex,vid)], key=lambda r:int(r['first_frame']))
            people = {r['person_id'] for r in gt}
            if len(people) != 1:
                raise ValueError(f'Multiple annotated subjects: {key}')
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {p}')
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
            if n <= 0 or not np.isfinite(fps) or fps <= 0 or min(w,h) <= 0:
                raise ValueError(f'Invalid video metadata: {p}')
            for r in gt:
                if not 0 <= int(r['first_frame']) <= int(r['last_frame']) < n:
                    raise ValueError(f'GT range outside video: {key}, {r}')
            legacy = ex == 6 and cam == 17
            v = dict(sample_id=f'Ex{ex}_Camera{cam}_{vid}', exercise_id=ex, exercise=NAMES[ex], camera=cam, video_id=vid, person_id=next(iter(people)), path=str(p.resolve()), frames=n, fps=fps, width=w, height=h, legacy=legacy, cache_dir=str(LEGACY if legacy else NEW / f'Ex{ex}' / f'Camera{cam}'), reps=len(gt), annotations=gt)
            v['cache_status'] = check_cache(v,model_hash,verify_video=True) if legacy else check_cache(v,model_hash)
            if legacy and v['cache_status'] != 'complete':
                raise RuntimeError(f'Legacy cache not compatible: {v["sample_id"]}: {v["cache_status"]}')
            rows.append(v)
    expected = {(ex,vid,cam) for ex,vid in groups for cam in (17,18)}
    if seen != expected or len(rows) != 130:
        raise ValueError(f'Video inventory mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}, total={len(rows)}')
    save_json(WORK/'manifest.json', {'weights':str(weights.resolve()), 'weights_sha256':model_hash, 'videos':rows})
    print('Manifest:', WORK/'manifest.json')
    for ex in NAMES:
        subset = [v for v in rows if v['exercise_id']==ex]
        print(f'Ex{ex}: videos={len(subset)} reps_one_view={sum(v["reps"] for v in subset if v["camera"]==17)} frames={sum(v["frames"] for v in subset)} complete={sum(v["cache_status"]=="complete" for v in subset)}')
    print('Total:', len(rows), 'videos,', sum(v['frames'] for v in rows), 'frames')


def select_videos(rows,args):
    rows = [v for v in rows if not v['legacy']]
    if args.pilot:
        selected=[]
        for ex,cam in [(6,18),(3,17),(3,18),(5,17)]:
            selected.append(next(v for v in rows if v['exercise_id']==ex and v['camera']==cam))
        rows=selected
    if args.sample:
        unknown=set(args.sample)-{v['sample_id'] for v in rows}
        if unknown: raise ValueError(f'Unknown sample(s): {unknown}')
        rows=[v for v in rows if v['sample_id'] in args.sample]
    return rows


def read_frame(cap,i):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok, frame=cap.read()
    if not ok: raise RuntimeError(f'Cannot decode frame {i}')
    return frame


def show_windows(path):
    try:
        win=subprocess.check_output(['wslpath','-w',str(path.resolve())],text=True).strip()
        subprocess.Popen(['explorer.exe',win],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.SubprocessError): pass


def image_write(path,img):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),img): raise RuntimeError(f'Cannot write {path}')


def review(rows,backend,anchors,no_open):
    for v in rows:
        key=v['sample_id']
        if key in anchors:
            print(key, 'already reviewed'); continue
        gt=v['annotations']; first=gt[0]; mid=gt[len(gt)//2]
        ids=sorted({int(first['first_frame']), (int(first['first_frame'])+int(first['last_frame']))//2, (int(mid['first_frame'])+int(mid['last_frame']))//2})
        cap=cv2.VideoCapture(v['path']); images=[]; candidates={}
        try:
            for i in ids:
                frame=read_frame(cap,i); boxes,scores,kps=backend.predict(frame)
                for j,(box,score) in enumerate(zip(boxes,scores)):
                    x1,y1,x2,y2=box.astype(int); color=(0,220,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,4)
                    cv2.putText(frame,f'{i}:{j} {score:.2f}',(max(0,x1),max(30,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
                    if kps is not None:
                        candidates[f'{i}:{j}']={'bbox':box.tolist(),'anchor_frame':i,'selected_id':j,'frame_width':v['width'],'frame_height':v['height'],'video_path':v['path']}
                cv2.putText(frame,f'{key} frame {i}',(12,36),cv2.FONT_HERSHEY_SIMPLEX,1,(0,220,255),3)
                images.append(cv2.resize(frame,(int(round(frame.shape[1]*600/frame.shape[0])),600)))
        finally: cap.release()
        path=WORK/'previews'/f'{key}.jpg'; image_write(path,np.vstack(images))
        print(f'\n{key}: {path.resolve()}\nCandidates: {", ".join(candidates)}')
        if not no_open: show_windows(path)
        if not candidates:
            print('No candidates; skipped.'); continue
        while True:
            ans=input('Choose correct subject frame:ID, s=skip, q=quit: ').strip()
            if ans=='q': return
            if ans=='s': break
            if ans in candidates:
                anchors[key]=candidates[ans]; save_json(WORK/'anchors.json',anchors); break
            print('Invalid choice; use a displayed frame:ID.')


def extract(rows,backend,anchors,model_hash):
    from extract_video_kp_cache import extract_video
    for v in rows:
        key=v['sample_id']
        if key not in anchors:
            print(key,'SKIP: not reviewed'); continue
        anchor=anchors[key]
        if anchor['video_path']!=v['path']: raise ValueError(f'Anchor path mismatch: {key}')
        status=check_cache(v,model_hash,verify_video=True,verify_anchor=anchor)
        if status=='complete': print(key,'reuse complete cache'); continue
        if status!='missing': raise RuntimeError(f'{key}: {status}; inspect files before retrying (no automatic overwrite)')
        print('\n===',key,'===',flush=True)
        meta=extract_video(v['video_id'],v['path'],anchor,backend,Path(v['cache_dir']),anchor_config_path=WORK/'anchors.json')
        if not meta['complete'] or check_cache(v,model_hash,verify_anchor=anchor)!='complete':
            raise RuntimeError(f'{key}: extraction incomplete')


def audit(rows,model_hash):
    from extract_video_kp_cache import KP_DUMP_DTYPE
    report=[]
    for v in rows:
        status=check_cache(v,model_hash)
        if status!='complete':
            print(v['sample_id'],status); continue
        paths=cache_paths(v); data=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r'); n=len(data)
        with np.load(paths[2],allow_pickle=False) as z:
            sel=z['status']==0; multi=z['person_count']>1; boxes=z['selected_box'].copy()
        mask=np.zeros(n,bool)
        for r in v['annotations']: mask[int(r['first_frame']):int(r['last_frame'])+1]=True
        kp=data['kp']; valid=np.isfinite(kp[:,:,:2]).all(axis=2)&(kp[:,:,2]>=.3)
        centers=(boxes[:,:2]+boxes[:,2:])/2; size=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
        jumps=np.linalg.norm(np.diff(centers,axis=0),axis=1)/size[:-1]; adjacent=sel[1:]&sel[:-1]
        max_jump=float(np.max(jumps[adjacent])) if adjacent.any() else None
        row=dict(sample_id=v['sample_id'],frames=n,selected_pct=round(100*sel.mean(),2),gt_selected_pct=round(100*sel[mask].mean(),2),gt_valid_kp_pct=round(100*valid[mask].mean(),2),multi_person_frames=int(multi.sum()),max_box_jump_ratio=round(max_jump,3) if max_jump is not None else None)
        report.append(row); print(json.dumps(row,ensure_ascii=False))
        # Review sampled overlays, including the largest consecutive selected-box jump.
        ids={int(v['annotations'][0]['first_frame']),int(v['annotations'][len(v['annotations'])//2]['first_frame']),int(v['annotations'][-1]['last_frame'])}
        if adjacent.any(): ids.add(int(np.argmax(np.where(adjacent,jumps,-1)))+1)
        multi_idx=np.flatnonzero(mask&multi)
        if len(multi_idx): ids.add(int(multi_idx[len(multi_idx)//2]))
        cap=cv2.VideoCapture(v['path']); images=[]
        try:
            for i in sorted(ids):
                frame=read_frame(cap,i)
                if sel[i]:
                    x1,y1,x2,y2=boxes[i].astype(int); cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),4)
                    for x,y in kp[i,valid[i],:2]: cv2.circle(frame,(int(x),int(y)),5,(0,220,255),-1)
                cv2.putText(frame,f'{v["sample_id"]} frame {i} selected={int(sel[i])}',(12,36),cv2.FONT_HERSHEY_SIMPLEX,1,(0,220,255),3)
                images.append(cv2.resize(frame,(int(round(frame.shape[1]*600/frame.shape[0])),600)))
        finally: cap.release()
        image_write(WORK/'qa'/f'{v["sample_id"]}.jpg',np.vstack(images))
    if report:
        path=WORK/'qa'/ 'audit.csv'; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(report[0])); w.writeheader(); w.writerows(report)
        print('QA:',path)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('mode',choices=['manifest','review','extract','audit']); ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--weights',type=Path,default=WEIGHTS); ap.add_argument('--pilot',action='store_true'); ap.add_argument('--sample',action='append'); ap.add_argument('--no-open',action='store_true'); args=ap.parse_args()
    if args.mode=='manifest': build_manifest(args.root,args.weights); return
    from extract_video_kp_cache import UltralyticsPoseBackend,sha256_file
    manifest=load_json(WORK/'manifest.json'); model_hash=manifest['weights_sha256']; weights=Path(manifest['weights'])
    if sha256_file(weights)!=model_hash: raise ValueError('Model weights changed; regenerate manifest')
    rows=select_videos(manifest['videos'],args)
    if args.mode=='audit': audit(rows,model_hash); return
    anchor_path=WORK/'anchors.json'; anchors=load_json(anchor_path) if anchor_path.exists() else {}
    backend=UltralyticsPoseBackend(str(weights),device='0',imgsz=640,conf=.25)
    if args.mode=='review': review(rows,backend,anchors,args.no_open)
    else: extract(rows,backend,anchors,model_hash)

if __name__=='__main__': main()
