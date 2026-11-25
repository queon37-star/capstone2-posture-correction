#!/usr/bin/env python3
"""Resume REHAB24 preprocessing with conservative auto anchors and QA.
Place beside preprocess_rehab24.py (v2) and test_rehab24_identity.py.
Run from pose_training_offline. Never overwrite existing cache files.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import cv2
import preprocess_rehab24 as pre

SPECIAL = 'Ex3_Camera17_PM_010'
BATCH = pre.WORK / 'batch_manifest.json'
ANCHORS = pre.WORK / 'anchors.json'


def prepare(run):
    manifest = pre.load_json(pre.WORK / 'manifest.json')
    v = next(v for v in manifest['videos'] if v['sample_id'] == SPECIAL)
    comp = pre.load_json(run / 'comparison.json')
    if comp['sample'] != SPECIAL or comp['anchor_x_limit'] != .6 or comp['trial']['gt_selected_pct'] != 100.0 or comp['trial']['selected'] != 4065:
        raise ValueError('Not the reviewed identity trial')
    anchor = pre.load_json(ANCHORS)[SPECIAL]
    source = dict(v, cache_dir=str((run / 'cache').resolve()))
    if pre.check_cache(source, manifest['weights_sha256'], verify_video=True, verify_anchor=anchor) != 'complete':
        raise ValueError('Trial cache missing or incompatible')
    v['cache_dir'] = str(pre.NEW / 'identity_reviewed' / 'Ex3' / 'Camera17')
    v['training_exclude_outside_gt'] = True
    v['identity_review'] = 'Sampled review; identity switches observed outside GT; GT identity not exhaustively verified'
    v['cache_provenance'] = str(run.resolve())
    dst = pre.cache_paths(v)
    src = pre.cache_paths(source)
    if not any(p.exists() for p in dst):
        dst[0].parent.mkdir(parents=True, exist_ok=True)
        for s,d in zip(src,dst):
            shutil.copy2(s,d)
    else:
        from extract_video_kp_cache import sha256_file
        if not all(d.is_file() and sha256_file(s)==sha256_file(d) for s,d in zip(src,dst)):
            raise RuntimeError('Existing reviewed cache differs; no overwrite')
    if pre.check_cache(v, manifest['weights_sha256'], verify_video=True, verify_anchor=anchor) != 'complete':
        raise RuntimeError('Reviewed cache validation failed')
    manifest['unannotated_policy'] = 'Unknown; do not label as Idle without separate review'
    pre.save_json(BATCH, manifest)
    print('Batch manifest:', BATCH, '\nReviewed cache:', v['cache_dir'])


def extra_flag(v):
    rows = v['annotations'] + [r['annotation'] for r in v.get('excluded_annotations', [])]
    return max(int(r[f'extra_person_in_cam{v["camera"]}']) for r in rows)


def choose_auto(v, backend):
    if extra_flag(v) != 0:
        return None, 'extra_person_annotation'
    gt = v['annotations']
    ids = sorted({(int(gt[j]['first_frame'])+int(gt[j]['last_frame']))//2
                  for j in np.linspace(0,len(gt)-1,min(5,len(gt)),dtype=int)})
    cap = cv2.VideoCapture(v['path'])
    candidates=[]
    try:
        for i in ids:
            frame=pre.read_frame(cap,i)
            boxes,scores,kps=backend.predict(frame)
            if len(boxes)!=1 or kps is None or kps.shape!=(1,17,3):
                return None,'not_single_person_in_all_probes'
            b=boxes[0]; kp=kps[0]
            if not np.isfinite(b).all() or np.any(b[2:]<=b[:2]) or scores[0]<.5 or np.sum(np.isfinite(kp[:,:2]).all(axis=1)&(kp[:,2]>=.3))<10:
                return None,'weak_probe'
            candidates.append((i,b.copy()))
    finally:
        cap.release()
    boxes=np.stack([b for _,b in candidates]); centers=(boxes[:,:2]+boxes[:,2:])/2
    scale=np.maximum(np.median(boxes[:,2:]-boxes[:,:2],axis=0),1)
    dist=np.linalg.norm((centers-np.median(centers,axis=0))/scale,axis=1)
    if np.max(dist)>.45:
        return None,'inconsistent_probe_centers'
    i,b=candidates[int(np.argmin(dist))]
    return dict(bbox=b.tolist(),anchor_frame=i,selected_id=0,frame_width=v['width'],frame_height=v['height'],video_path=v['path'],selection_method='auto_single_subject',probe_frames=ids), 'auto'


def auto_review(rows,backend,anchors):
    pending=[]; counts={'existing':0,'auto':0,'manual':0}
    for v in rows:
        key=v['sample_id']
        if key in anchors:
            counts['existing']+=1; continue
        a,reason=choose_auto(v,backend)
        if a is not None:
            anchors[key]=a; pre.save_json(ANCHORS,anchors); counts['auto']+=1
        else:
            pending.append(dict(sample_id=key,reason=reason)); counts['manual']+=1
        print(key,reason,flush=True)
    pre.save_json(pre.WORK/'manual_review_queue.json',pending)
    print(json.dumps(counts,indent=2))


def extract(rows,backend,anchors,model_hash):
    from extract_video_kp_cache import extract_video
    import extract_all_squats_tracked as tracked
    from test_rehab24_identity import Recovery, Backend
    original=tracked.select_target
    for v in rows:
        key=v['sample_id']; a=anchors.get(key)
        if a is None:
            print(key,'SKIP: anchor missing'); continue
        if a['video_path']!=v['path']:
            raise ValueError(f'{key}: anchor path mismatch')
        status=pre.check_cache(v,model_hash,verify_video=True,verify_anchor=a)
        if status=='complete':
            print(key,'reuse'); continue
        if status!='missing':
            print(key,'SKIP:',status,'(no overwrite)'); continue
        gap=max(1,int(np.ceil(.5*v['fps'])))
        recovery=Recovery(original,gap)
        try:
            tracked.select_target=recovery
            meta=extract_video(v['video_id'],v['path'],a,Backend(backend,recovery),Path(v['cache_dir']),anchor_config_path=ANCHORS)
        finally:
            tracked.select_target=original
        meta['target_selection'].update(rule='original ROI + single-person reacquisition',anchor_x_limit=.95,gap_frames=gap,recovery_events=len(recovery.events))
        pre.save_json(pre.cache_paths(v)[1],meta)
        if pre.check_cache(v,model_hash,verify_anchor=a)!='complete':
            raise RuntimeError(f'{key}: incomplete extraction')
        pre.save_json(pre.WORK/'recovery_events'/f'{key}.json',recovery.events)


def audit(rows,model_hash):
    # Existing audit produces sampled overlays and per-video metrics.
    pre.audit(rows,model_hash)
    flags=[]; completed=0
    for v in rows:
        key=v['sample_id']; state=pre.check_cache(v,model_hash)
        if state!='complete':
            flags.append(dict(sample_id=key,reason=state)); continue
        completed+=1
        with np.load(pre.cache_paths(v)[2],allow_pickle=False) as z:
            status=z['status']; boxes=z['selected_box']; persons=z['person_count']
        gt=np.zeros(len(status),bool)
        for r in v['annotations']:
            gt[int(r['first_frame']):int(r['last_frame'])+1]=True
        sel=status==0
        centers=(boxes[:,:2]+boxes[:,2:])/2
        scale=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
        jumps=np.linalg.norm(np.diff(centers,axis=0),axis=1)/scale[:-1]
        adjacent=sel[:-1]&sel[1:]
        ids=np.flatnonzero(adjacent & (jumps>=.25))
        reasons=[]
        if np.mean(sel[gt])<.99: reasons.append('gt_selection_below_99pct')
        if np.any(persons>1): reasons.append('multiple_detections')
        if extra_flag(v)>0: reasons.append('extra_person_annotation')
        if len(ids): reasons.append('large_box_jump')
        if key==SPECIAL: reasons.append('known_identity_switch_outside_gt')
        if reasons:
            flags.append(dict(sample_id=key,reason=','.join(reasons),gt_selected_pct=round(float(100*np.mean(sel[gt])),3),large_jump_pairs=[[int(i),int(i+1)] for i in ids],gt_large_jump_pairs=[[int(i),int(i+1)] for i in ids if gt[i] or gt[i+1]]))
    pre.save_json(pre.WORK/'batch_review_flags.json',flags)
    print(f'Complete: {completed}/{len(rows)}; flagged: {len(flags)}; report: {pre.WORK/"batch_review_flags.json"}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['prepare','auto','review','extract','audit'])
    ap.add_argument('--identity-run',type=Path)
    ap.add_argument('--exercise',type=int,action='append',choices=range(1,7))
    ap.add_argument('--camera',type=int,choices=[17,18])
    ap.add_argument('--no-open',action='store_true')
    args=ap.parse_args()
    if args.mode=='prepare':
        if args.identity_run is None: ap.error('prepare requires --identity-run')
        prepare(args.identity_run); return
    manifest=pre.load_json(BATCH); model_hash=manifest['weights_sha256']
    from extract_video_kp_cache import UltralyticsPoseBackend, sha256_file
    if sha256_file(manifest['weights'])!=model_hash: raise ValueError('Weights changed')
    rows=[v for v in manifest['videos'] if not v['legacy'] and (not args.exercise or v['exercise_id'] in args.exercise) and (not args.camera or v['camera']==args.camera)]
    anchors=pre.load_json(ANCHORS) if ANCHORS.exists() else {}
    if args.mode=='audit': audit(rows,model_hash); return
    backend=UltralyticsPoseBackend(manifest['weights'],device='0',imgsz=640,conf=.25)
    if args.mode=='auto': auto_review(rows,backend,anchors)
    elif args.mode=='review': pre.review(rows,backend,anchors,args.no_open)
    else: extract(rows,backend,anchors,model_hash)

if __name__=='__main__': main()
