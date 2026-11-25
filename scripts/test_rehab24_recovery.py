#!/usr/bin/env python3
"""Isolated tracking recovery experiment; run from pose_training_offline/scripts parent."""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import preprocess_rehab24 as pre
import extract_all_squats_tracked as tracked
from extract_video_kp_cache import UltralyticsPoseBackend, extract_video, sha256_file

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
    args=ap.parse_args()
    if args.gap_sec<=0: ap.error('--gap-sec must be positive')
    manifest=pre.load_json(pre.WORK/'manifest.json')
    v=next(x for x in manifest['videos'] if x['sample_id']==args.sample)
    if v['legacy']: raise ValueError('Legacy caches are not part of this experiment')
    anchors=pre.load_json(pre.WORK/'anchors.json'); anchor=anchors[args.sample]
    model_hash=manifest['weights_sha256']
    if sha256_file(manifest['weights'])!=model_hash: raise ValueError('Weights hash mismatch')
    if pre.check_cache(v,model_hash,verify_video=True,verify_anchor=anchor)!='complete': raise ValueError('Baseline cache missing/incompatible')
    out=Path('results/rehab24_tracking_recovery')/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.sample)
    out.mkdir(parents=True,exist_ok=False)
    trial=dict(v,cache_dir=str(out/'cache'))
    original=tracked.select_target
    gap=max(1,int(np.ceil(args.gap_sec*v['fps'])))
    recovery=Recovery(original,gap)
    backend=Backend(UltralyticsPoseBackend(manifest['weights'],device='0',imgsz=640,conf=.25),recovery)
    try:
        tracked.select_target=recovery
        meta=extract_video(v['video_id'],v['path'],anchor,backend,Path(trial['cache_dir']),anchor_config_path=pre.WORK/'anchors.json')
    finally:
        tracked.select_target=original
    paths=pre.cache_paths(trial)
    meta['target_selection']['rule']='original select_target + single-person anchor-ROI reacquisition after gap'
    meta['target_selection']['gap_frames']=gap
    meta['target_selection']['gap_sec_requested']=args.gap_sec
    meta['target_selection']['recovery_events']=len(recovery.events)
    pre.save_json(paths[1],meta)
    if pre.check_cache(trial,model_hash,verify_video=True,verify_anchor=anchor)!='complete': raise RuntimeError('Trial cache incomplete')
    result=dict(sample=args.sample,baseline=stats(v,pre.cache_paths(v)[2]),trial=stats(v,paths[2]),rejection_counts=recovery.rejected,recovery_events=recovery.events)
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
    print('Results:',out.resolve())
    print('Original cache unchanged; trial not promoted.')

if __name__=='__main__': main()
