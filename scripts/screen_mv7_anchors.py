#!/usr/bin/env python3
"""Conservative nine-frame MV7 anchor screening. Run from pose_training_offline.
Preserves existing anchors/caches. Deferred videos are not deleted or rejected permanently.
Requires preprocess_extra_datasets.py (v2), preprocess_rehab24.py, repo pose backend.
"""
import argparse
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import preprocess_extra_datasets as extra
import preprocess_rehab24 as pre

WORK = extra.WORK
CONF = .30
CHAINS = {
    'squat': ((5,11,13,15), (6,12,14,16)),
    'push_up': ((5,7,9,11,13,15), (6,8,10,12,14,16)),
}


def joint_valid(kp):
    kp = np.asarray(kp)
    return np.isfinite(kp[..., :2]).all(axis=-1) & np.isfinite(kp[..., 2]) & (kp[..., 2] >= CONF)


def chain_valid(valid, exercise):
    return np.logical_or.reduce([valid[..., list(c)].all(axis=-1) for c in CHAINS[exercise]])


def candidate(v, frame, j, box, score, kp):
    b = np.asarray(box, dtype=float)
    if not np.isfinite(b).all() or np.any(b[2:] <= b[:2]) or not np.isfinite(score):
        return None
    valid = joint_valid(kp)
    body = int(valid[5:17].sum())
    chain = bool(chain_valid(valid, v['exercise']))
    return dict(bbox=b.tolist(), anchor_frame=int(frame), selected_id=int(j),
                frame_width=v['width'], frame_height=v['height'], video_path=v['path'],
                score=float(score), valid_kp=int(valid.sum()), body_valid=body,
                chain_valid=chain, strong=bool(score >= .50 and body >= 6 and chain))


def classify(probes):
    """No anchor if multiple plausible subjects, weak probes, or inconsistent centers."""
    strong = [[c for c in p['candidates'] if c['strong']] for p in probes]
    n = len(probes)
    good = sum(bool(s) for s in strong)
    detail = dict(probe_count=n, strong_probe_count=good,
                  ambiguous_probe_count=sum(len(s)>1 for s in strong))
    if not n or good < int(np.ceil(n*2/3)):
        return 'defer', 'insufficient_body_chain_probes', None, detail
    if any(len(s)!=1 for s in strong):
        return 'review', 'missing_or_multiple_strong_candidates', None, detail
    chosen = [s[0] for s in strong]
    # Another similarly sized, confident person with some body joints can be a plausible target,
    # even if its full joint chain is occluded.
    for probe, target in zip(probes, chosen):
        b = np.asarray(target['bbox']); area = np.prod(b[2:]-b[:2])
        for c in probe['candidates']:
            if c['selected_id'] == target['selected_id']:
                continue
            b2 = np.asarray(c['bbox']); area2 = np.prod(b2[2:]-b2[:2])
            if c['score'] >= .50 and c['body_valid'] >= 4 and area2 >= .5 * area:
                return 'review', 'plausible_second_person', None, detail
    boxes = np.asarray([c['bbox'] for c in chosen]); sizes=boxes[:,2:]-boxes[:,:2]
    centers=(boxes[:,:2]+boxes[:,2:])/2
    scale=max(float(np.median(np.linalg.norm(sizes,axis=1))),1.)
    dist=np.linalg.norm(centers-np.median(centers,axis=0),axis=1)/scale
    detail['max_center_deviation']=float(dist.max())
    if dist.max()>.35:
        return 'review', 'inconsistent_centers', None, detail
    areas=np.prod(sizes,axis=1); med=float(np.median(areas))
    if areas.min()<.35*med or areas.max()>2.5*med:
        return 'review', 'inconsistent_box_area', None, detail
    j=min(range(n),key=lambda i:(-chosen[i]['body_valid'],float(dist[i]),-chosen[i]['score']))
    a={k:v for k,v in chosen[j].items() if k!='strong'}
    a.update(selection_method='auto_mv7_body_chain_v1',probe_frames=[p['frame'] for p in probes])
    return 'auto','unique_strong_consistent_candidate',a,detail


def inspect(v,backend):
    ids=sorted({int(round((v['frames']-1)*q)) for q in np.linspace(.1,.9,9)})
    cap=cv2.VideoCapture(v['path']); probes=[]
    try:
        for i in ids:
            frame=pre.read_frame(cap,i); boxes,scores,kps=backend.predict(frame)
            candidates=[]
            if kps is not None:
                for j,(b,score,kp) in enumerate(zip(boxes,scores,kps)):
                    c=candidate(v,i,j,b,score,kp)
                    if c is not None: candidates.append(c)
            probes.append(dict(frame=i, detections=len(boxes), candidates=candidates))
    finally: cap.release()
    return probes


def cache_quality(v):
    from extract_video_kp_cache import KP_DUMP_DTYPE
    data=np.memmap(pre.cache_paths(v)[0],dtype=KP_DUMP_DTYPE,mode='r')
    valid=joint_valid(data['kp']); body=valid[:,5:17]
    chain=chain_valid(valid,v['exercise'])
    with np.load(pre.cache_paths(v)[2],allow_pickle=False) as z: selected=z['status']==0
    chain &= selected
    edges=np.flatnonzero(np.diff(np.r_[False,~chain,False])); gaps=edges[1::2]-edges[::2]
    return dict(selected_pct=round(100*float(selected.mean()),3),
                body_valid_pct=round(100*float(body.mean()),3),
                chain_valid_pct=round(100*float(chain.mean()),3),
                longest_chain_gap_sec=round(float(gaps.max())/v['fps'],3) if len(gaps) else 0.)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exercise',action='append',choices=sorted(CHAINS))
    ap.add_argument('--subject',action='append',help='e.g. subject_002; default all subjects')
    ap.add_argument('--dry-run',action='store_true',help='report only; do not register new anchors')
    args=ap.parse_args()
    from extract_video_kp_cache import UltralyticsPoseBackend,sha256_file
    m=pre.load_json(WORK/'manifest.json'); mh=m['weights_sha256']
    if sha256_file(m['weights'])!=mh: raise ValueError('Weights changed')
    rows=[v for v in m['videos'] if v['dataset']=='mv7' and v['exercise'] in (args.exercise or CHAINS)
          and (not args.subject or v['person_id'] in args.subject)]
    if not rows: ap.error('No matching MV7 videos')
    path=WORK/'anchors.json'; anchors=pre.load_json(path) if path.exists() else {}
    backend=UltralyticsPoseBackend(m['weights'],device='0',imgsz=640,conf=.25)
    records=[]; counts=Counter(); by_ex={e:Counter() for e in CHAINS}
    report_path=WORK/'mv7_body_screening.json'
    # Write progressively so interruption does not lose completed results; scope is explicit.
    def save():
        pre.save_json(report_path,dict(scope=dict(exercise=args.exercise,subject=args.subject,dry_run=args.dry_run),
                     total=len(rows),processed=len(records),counts=dict(counts),
                     by_exercise={e:dict(c) for e,c in by_ex.items()},records=records))
    for v in rows:
        sid=v['sample_id']; a=anchors.get(sid)
        r=dict(sample_id=sid,person_id=v['person_id'],exercise=v['exercise'],camera=v['camera'],clip_correctness=v['clip_correctness'])
        try:
            state=pre.check_cache(v,mh,verify_anchor=a)
            if state=='complete':
                r.update(status='existing',reason='complete_cache_preserved',cache_quality=cache_quality(v))
            elif state!='missing':
                r.update(status='review',reason='cache_'+state)
            elif a is not None:
                r.update(status='existing',reason='existing_anchor_preserved')
            else:
                probes=inspect(v,backend); status,reason,new_anchor,detail=classify(probes)
                r.update(status=status,reason=reason,**detail,probes=probes)
                if new_anchor is not None and not args.dry_run:
                    anchors[sid]=new_anchor; pre.save_json(path,anchors)
        except (RuntimeError,ValueError,OSError,cv2.error) as exc:
            r.update(status='error',reason=str(exc))
        records.append(r); counts[r['status']]+=1; by_ex[v['exercise']][r['status']]+=1
        print(f"{len(records)}/{len(rows)} {sid} {r['status']} {r['reason']}",flush=True)
        save()
    print('Summary:',dict(counts)); print('By exercise:',{e:dict(c) for e,c in by_ex.items()})
    print('Report:',report_path)

if __name__=='__main__': main()
