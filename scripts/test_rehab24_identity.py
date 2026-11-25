#!/usr/bin/env python3
"""Isolated tracking recovery experiment; run from pose_training_offline/scripts parent."""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import preprocess_rehab24 as pre
import extract_all_squats_tracked as tracked
from extract_video_kp_cache import UltralyticsPoseBackend, extract_video, sha256_file

def restrict_horizontal(original, limit):
    """Keep original detection indices; only reject candidates far from anchor x."""
    def select(boxes, scores, anchor, prev):
        if not len(boxes):
            return None
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        ac = (anchor[0] + anchor[2]) / 2
        width = max(float(anchor[2] - anchor[0]), 1.0)
        keep = np.flatnonzero(np.abs(cx-ac)/width <= limit)
        if not len(keep):
            return None
        i = original(boxes[keep], scores[keep], anchor, prev)
        return None if i is None else int(keep[i])
    return select


def extra_qa(v, paths, out, events):
    with np.load(paths[2], allow_pickle=False) as z:
        status = z['status'].copy(); boxes = z['selected_box'].copy()
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    diag = np.maximum(np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1), 1)
    jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1) / diag[:-1]
    adjacent = (status[:-1] == 0) & (status[1:] == 0)
    ranked = np.flatnonzero(adjacent & (jumps >= .25))
    ranked = sorted(ranked, key=lambda i: jumps[i], reverse=True)
    # Show both sides of the largest jumps, plus the previous failure/recovery area.
    ids = {230, 731, 1340, 1594, 1595, 1596, 1613, 2631, 3950}
    for i in ranked[:5]: ids.update([int(i), int(i+1)])
    for event in events: ids.update([event['frame']-1, event['frame']])
    gt = np.zeros(len(status), bool)
    for r in v['annotations']: gt[int(r['first_frame']):int(r['last_frame'])+1] = True
    cap = cv2.VideoCapture(v['path']); images = []
    try:
        for i in sorted(i for i in ids if 0 <= i < len(status)):
            frame = pre.read_frame(cap, i)
            if status[i] == 0:
                x1,y1,x2,y2 = boxes[i].astype(int)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),4)
            cv2.putText(frame,f'frame={i} selected={int(status[i]==0)} GT={int(gt[i])}',(12,36),cv2.FONT_HERSHEY_SIMPLEX,1,(0,220,255),3)
            images.append(cv2.resize(frame,(int(round(frame.shape[1]*360/frame.shape[0])),360)))
    finally: cap.release()
    if images: pre.image_write(out/'qa'/'identity_check.jpg',np.vstack(images))
    pre.save_json(out/'identity_diagnostics.json', {'large_adjacent_jumps': [{'from':int(i),'to':int(i+1),'ratio':float(jumps[i]),'gt_from':bool(gt[i]),'gt_to':bool(gt[i+1])} for i in ranked], 'reviewed_frame_ids': sorted(i for i in ids if 0 <= i < len(status))})


class Recovery:
    def __init__(self, original, gap_frames):
        self.original = original
        self.gap_frames = gap_frames
        self.frame = -1
        self.last_selected = None
        self.has_kp = False
        self.events = []
        self.rejected = {'roi': 0, 'continuity_or_combination': 0}

    def __call__(self, boxes, scores, anchor, prev):
        chosen = self.original(boxes, scores, anchor, prev)
        if chosen is None:
            roi_choice = self.original(boxes, scores, anchor, None)
            reason = 'roi' if roi_choice is None else 'continuity_or_combination'
            self.rejected[reason] += 1
            gap = self.frame - self.last_selected if self.last_selected is not None else self.frame + 1
            if (len(boxes) == 1 and roi_choice is not None and self.has_kp
                    and gap >= self.gap_frames):
                chosen = roi_choice
                self.events.append(dict(frame=self.frame, gap_frames=gap, reason=reason,
                                        previous_box=None if prev is None else prev.tolist(),
                                        selected_box=boxes[chosen].tolist()))
        if chosen is not None and self.has_kp:
            self.last_selected = self.frame
        return chosen

class Backend:
    def __init__(self, backend, recovery):
        self.backend = backend
        self.recovery = recovery
    def predict(self, frame):
        self.recovery.frame += 1
        boxes, scores, kps = self.backend.predict(frame)
        self.recovery.has_kp = kps is not None
        return boxes, scores, kps
    def describe(self):
        return self.backend.describe()

def stats(v, path):
    with np.load(path, allow_pickle=False) as z:
        status = z['status'].copy(); boxes=z['selected_box'].copy(); persons=z['person_count'].copy()
    selected=status==0
    gt=np.zeros(len(status),bool)
    for r in v['annotations']:
        gt[int(r['first_frame']):int(r['last_frame'])+1]=True
    edges=np.flatnonzero(np.diff(np.r_[False,~selected,False]))
    gaps=edges[1::2]-edges[::2]
    centers=(boxes[:,:2]+boxes[:,2:])/2
    diag=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
    jumps=np.linalg.norm(np.diff(centers,axis=0),axis=1)/diag[:-1]
    adjacent=selected[:-1]&selected[1:]
    return dict(frames=len(status),selected=int(selected.sum()),selected_pct=round(100*selected.mean(),2),gt_selected_pct=round(100*selected[gt].mean(),2),no_detection=int((status==1).sum()),no_target=int((status==2).sum()),no_keypoints=int((status==3).sum()),longest_failure=int(gaps.max()) if len(gaps) else 0,max_box_jump_ratio=round(float(jumps[adjacent].max()),3) if adjacent.any() else None,multi_person_frames=int((persons>1).sum()))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample',default='Ex3_Camera17_PM_010')
    ap.add_argument('--gap-sec',type=float,default=.5)
    ap.add_argument('--anchor-x-limit',type=float,default=.60)
    args=ap.parse_args()
    if args.gap_sec<=0: ap.error('--gap-sec must be positive')
    if not 0 < args.anchor_x_limit <= .95: ap.error('--anchor-x-limit must be in (0, .95]')
    if args.sample != 'Ex3_Camera17_PM_010': ap.error('This pilot is specific to PM_010 Camera17')
    manifest=pre.load_json(pre.WORK/'manifest.json')
    v=next(x for x in manifest['videos'] if x['sample_id']==args.sample)
    if v['legacy']: raise ValueError('Legacy caches are not part of this experiment')
    anchors=pre.load_json(pre.WORK/'anchors.json'); anchor=anchors[args.sample]
    model_hash=manifest['weights_sha256']
    if sha256_file(manifest['weights'])!=model_hash: raise ValueError('Weights hash mismatch')
    if pre.check_cache(v,model_hash,verify_video=True,verify_anchor=anchor)!='complete': raise ValueError('Baseline cache missing/incompatible')
    out=Path('results/rehab24_tracking_identity')/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.sample)
    out.mkdir(parents=True,exist_ok=False)
    trial=dict(v,cache_dir=str(out/'cache'))
    original=tracked.select_target
    gap=max(1,int(np.ceil(args.gap_sec*v['fps'])))
    recovery=Recovery(restrict_horizontal(original,args.anchor_x_limit),gap)
    backend=Backend(UltralyticsPoseBackend(manifest['weights'],device='0',imgsz=640,conf=.25),recovery)
    try:
        tracked.select_target=recovery
        meta=extract_video(v['video_id'],v['path'],anchor,backend,Path(trial['cache_dir']),anchor_config_path=pre.WORK/'anchors.json')
    finally:
        tracked.select_target=original
    paths=pre.cache_paths(trial)
    meta['target_selection']['rule']='original select_target + narrower horizontal anchor ROI + single-person reacquisition after gap'
    meta['target_selection']['anchor_x_limit']=args.anchor_x_limit
    meta['target_selection']['gap_frames']=gap
    meta['target_selection']['gap_sec_requested']=args.gap_sec
    meta['target_selection']['recovery_events']=len(recovery.events)
    pre.save_json(paths[1],meta)
    if pre.check_cache(trial,model_hash,verify_video=True,verify_anchor=anchor)!='complete': raise RuntimeError('Trial cache incomplete')
    result=dict(sample=args.sample,anchor_x_limit=args.anchor_x_limit,baseline=stats(v,pre.cache_paths(v)[2]),trial=stats(v,paths[2]),rejection_counts=recovery.rejected,recovery_events=recovery.events)
    pre.save_json(out/'comparison.json',result)
    print('\n=== Baseline vs trial ===')
    print(json.dumps(result,indent=2))
    # The existing audit writes overlays and metrics. Redirect only its output root.
    old_work=pre.WORK
    try:
        pre.WORK=out
        pre.audit([trial],model_hash)
    finally:
        pre.WORK=old_work
    extra_qa(v,paths,out,recovery.events)
    print('Results:',out.resolve())
    print('Original cache unchanged; trial not promoted.')

if __name__=='__main__': main()
