#!/usr/bin/env python3
"""REHAB26 + Multi-View Seven preprocessing. Run from pose_training_offline.
Requires existing preprocess_rehab24.py (v2), test_rehab24_identity.py and repo extraction modules.
No REHAB24 files are modified. Video-level labels never become synthetic repetition GT.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import preprocess_rehab24 as pre

ROOT = Path.home() / 'capstone2/datasets'
WORK = Path('data/preprocess/extra_datasets')
CACHE = Path('data/cache/kp_full/extra_datasets_yolov8m')
WEIGHTS = Path.home() / 'capstone2/pose_training/yolov8m-pose.pt'
REHAB_NAMES = {1: 'arm_abduction', 6: 'squat', 7: 'shoulder_extension'}
MV_NAMES = {'abs','back','bicep_curl','push_up','shoulder','squat','tricep'}
VIEWS = {'front','side','diagonal'}

def exercise_class(name):
    return {'squat':'Squat','push_up':'Pushup'}.get(name,'Other')

def metadata(p):
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened(): raise ValueError(f'Cannot open {p}')
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS))
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n<=0 or not np.isfinite(fps) or fps<=0 or min(w,h)<=0: raise ValueError(f'Bad metadata {p}')
        return dict(frames=n,fps=fps,width=w,height=h)
    finally: cap.release()

def rehab_row(p, groups):
    m=re.fullmatch(r'(ex([167])_([^_]+))_cam([123])\.mp4',p.name)
    if not m: raise ValueError(f'Unexpected REHAB26 name {p}')
    vid, ex, person, cam=m[1],int(m[2]),m[3],int(m[4])
    if p.parent.name!=f'ex{ex}' or vid not in groups: raise ValueError(f'Unmatched REHAB26 video {p}')
    info=metadata(p); n=info['frames']
    if abs(info['fps']-30)>.01: raise ValueError(f'REHAB26 fps !=30: {p}')
    valid=[]; excluded=[]; seen=set(); last=-1
    for row in sorted(groups[vid],key=lambda r:int(r['first_frame'])):
        r=dict(row); start=int(r['first_frame']); end=int(r['last_frame']); orientation=int(r['person_orientation']); score=int(r['correctness_score'])
        if int(r['exercise_id'])!=ex or r['person']!=person or score not in (1,5) or orientation not in {0,10,20,30,40,45,50,60,70,80,90}: raise ValueError(f'Invalid annotation {r}')
        key=(orientation,int(r['repetition_number']))
        if key in seen: raise ValueError(f'Duplicate repetition {vid} {key}')
        seen.add(key)
        r.update(correctness=int(score==1), viewing_angle_deg={1:0,2:45,3:90}[cam]-orientation)
        if not 0<=start<=end<n or start<=last:
            excluded.append(dict(annotation=r,reason='invalid_or_overlapping_frame_range'))
        else: valid.append(r); last=end
    name=REHAB_NAMES[ex]; sid=f'Rehab26_{vid}_cam{cam}'
    return dict(sample_id=sid, dataset='rehab26', video_id=sid, recording_id=vid, person_id=person, group_id=f'rehab26:{person}',exercise=name,exercise_id=ex,exercise_class=exercise_class(name),camera=cam,path=str(p.resolve()),cache_dir=str(CACHE/'rehab26'/name),label_scope='repetition',annotations=valid,excluded_annotations=excluded,**info)

def mv_row(p, base):
    parts=p.relative_to(base).parts
    if len(parts)!=5: raise ValueError(f'Unexpected Multi-View path: {p}')
    subject,ex,quality,view,filename=parts
    if not re.fullmatch(r'subject_\d{3}',subject) or ex not in MV_NAMES or quality not in {'good','bad'} or view not in VIEWS or filename!=f'{subject}_{ex}_{quality}_{view}.mp4': raise ValueError(f'Inconsistent Multi-View path: {p}')
    sid=f'MV7_{p.stem}'
    return dict(sample_id=sid,dataset='mv7',video_id=sid,recording_id=f'{subject}_{ex}_{quality}',person_id=subject,group_id=f'mv7:{subject}',exercise=ex,exercise_class=exercise_class(ex),camera=view,path=str(p.resolve()),cache_dir=str(CACHE/'mv7'/ex),label_scope='video',clip_correctness=int(quality=='good'),annotations=[],excluded_annotations=[],**metadata(p))

def manifest(root,weights):
    from extract_video_kp_cache import sha256_file
    rroot=root/'REHAB26-ViewAngles'; mroot=root/'MultiView-Fitness-7'/'Dataset Subject-wise'
    groups=defaultdict(list)
    with (rroot/'Segmentation.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f,delimiter=';'): groups[r['video_id']].append(r)
    if len(groups)!=15 or sum(map(len,groups.values()))!=1665: raise ValueError('Unexpected REHAB26 annotation inventory')
    rows=[rehab_row(p,groups) for p in sorted((rroot/'videos').rglob('*.mp4'))]
    actual={(v['recording_id'],v['camera']) for v in rows}
    if len(rows)!=45 or actual!={(vid,cam) for vid in groups for cam in (1,2,3)}: raise ValueError('REHAB26 inventory mismatch')
    mv=[mv_row(p,mroot) for p in sorted(mroot.rglob('*.mp4'))]
    if len(mv)!=1042 or len({v['person_id'] for v in mv})!=26: raise ValueError(f'MV7 inventory mismatch: {len(mv)} videos')
    rows+=mv; mh=sha256_file(weights)
    if len({v['sample_id'] for v in rows})!=len(rows): raise ValueError('Duplicate sample ID')
    for v in rows: v['cache_status']=pre.check_cache(v,mh)
    exclusions=[dict(sample_id=v['sample_id'],**r) for v in rows for r in v['excluded_annotations']]
    pre.save_json(WORK/'annotation_exclusions.json',exclusions)
    pre.save_json(WORK/'manifest.json',dict(weights=str(weights.resolve()),weights_sha256=mh,videos=rows,unannotated_policy='Unknown; clip labels are weak labels, not repetition or frame GT'))
    for d in ('rehab26','mv7'):
        subset=[v for v in rows if v['dataset']==d]
        for ex in sorted({v['exercise'] for v in subset}):
            s=[v for v in subset if v['exercise']==ex]; print(d,ex,'videos=',len(s),'persons=',len({v['person_id'] for v in s}),'frames=',sum(v['frames'] for v in s),'reps_view=',sum(len(v['annotations']) for v in s),'clip_labels=',dict(Counter(v.get('clip_correctness') for v in s if v['label_scope']=='video')),'rep_labels=',dict(Counter(r['correctness'] for v in s for r in v['annotations'])))
    print('Total videos:',len(rows),'frames:',sum(v['frames'] for v in rows),'excluded camera-view annotations:',len(exclusions))
    print('Manifest:',WORK/'manifest.json')

def probes(v):
    if v['label_scope']=='repetition' and v['annotations']:
        gt=v['annotations']; return sorted({(int(gt[j]['first_frame'])+int(gt[j]['last_frame']))//2 for j in np.linspace(0,len(gt)-1,min(5,len(gt)),dtype=int)})
    return sorted({int(round((v['frames']-1)*q)) for q in (.1,.3,.5,.7,.9)})

def detect_probes(v, backend, preview=False):
    cap=cv2.VideoCapture(v['path']); candidates={}; images=[]; counts=[]; ids=probes(v)
    try:
        for i in ids:
            frame=pre.read_frame(cap,i); boxes,scores,kps=backend.predict(frame); counts.append(len(boxes))
            for j,(b,score) in enumerate(zip(boxes,scores)):
                if kps is None or not np.isfinite(b).all() or np.any(b[2:]<=b[:2]): continue
                kp=kps[j]; valid=int(np.sum(np.isfinite(kp[:,:2]).all(axis=1)&(kp[:,2]>=.3)))
                candidates[f'{i}:{j}']=dict(bbox=b.tolist(),anchor_frame=i,selected_id=j,frame_width=v['width'],frame_height=v['height'],video_path=v['path'],score=float(score),valid_kp=valid)
                if preview:
                    x1,y1,x2,y2=b.astype(int); cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,255),3); cv2.putText(frame,f'{i}:{j} {score:.2f}',(max(0,x1),max(25,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,220,255),2)
            if preview:
                cv2.putText(frame,f'frame {i}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,220,255),2)
                images.append(cv2.resize(frame,(max(1,round(frame.shape[1]*420/frame.shape[0])),420)))
    finally: cap.release()
    return candidates,counts,images,ids

def auto_choice(candidates,counts,ids):
    if not counts or any(n!=1 for n in counts) or len(candidates)!=len(ids): return None,'not_single_person_in_all_probes'
    c=list(candidates.values())
    if any(a['score']<.5 or a['valid_kp']<10 for a in c): return None,'weak_probe'
    boxes=np.asarray([a['bbox'] for a in c]); centers=(boxes[:,:2]+boxes[:,2:])/2
    scale=max(float(np.median(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1))),1.)
    dist=np.linalg.norm(centers-np.median(centers,axis=0),axis=1)/scale
    if max(dist)>.35: return None,'inconsistent_probe_centers'
    # Choose a real detection near the median center; favor high joint visibility.
    j=min(range(len(c)),key=lambda j:(-c[j]['valid_kp'],float(dist[j]),-c[j]['score']))
    return dict(c[j],selection_method='auto_single_subject',probe_frames=ids),'auto'

def auto_review(rows,backend,anchors,review,no_open):
    pending=[]; counts=Counter()
    for v in rows:
        sid=v['sample_id']
        if sid in anchors: counts['existing']+=1; continue
        try:
            c,counts_probe,images,ids=detect_probes(v,backend,preview=review)
        except RuntimeError as exc:
            print(sid,exc); pending.append(dict(sample_id=sid,reason=str(exc))); counts['manual']+=1; continue
        if not review:
            a,reason=auto_choice(c,counts_probe,ids)
        else:
            path=WORK/'previews'/f'{sid}.jpg'
            if images: pre.image_write(path,np.vstack(images))
            print(sid,path.resolve(),'Candidates:',', '.join(c))
            if not no_open and images: pre.show_windows(path)
            a=None; reason='manual_skipped'
            if c:
                while True:
                    ans=input('Choose frame:ID, s=skip, q=quit: ').strip()
                    if ans=='q': return
                    if ans=='s': break
                    if ans in c: a=dict(c[ans],selection_method='manual'); reason='manual_selected'; break
                    print('Invalid choice')
        if a is not None:
            anchors[sid]=a; pre.save_json(WORK/'anchors.json',anchors); counts[reason]+=1
        else: pending.append(dict(sample_id=sid,reason=reason)); counts['manual']+=1
        print(sid,reason,flush=True)
    pre.save_json(WORK/('manual_remaining.json' if review else 'manual_review_queue.json'),pending)
    print(dict(counts))

def extract(rows,backend,anchors,mh):
    import extract_all_squats_tracked as tracked
    from extract_video_kp_cache import extract_video
    from test_rehab24_identity import Recovery,Backend
    original=tracked.select_target
    for v in rows:
        sid=v['sample_id']; a=anchors.get(sid)
        if a is None: print(sid,'SKIP no anchor'); continue
        if a['video_path']!=v['path']: raise ValueError('Anchor path mismatch '+sid)
        state=pre.check_cache(v,mh,verify_video=True,verify_anchor=a)
        if state!='missing': print(sid,state,'(no overwrite)'); continue
        gap=max(1,int(np.ceil(.5*v['fps']))); rec=Recovery(original,gap)
        try:
            tracked.select_target=rec
            meta=extract_video(v['video_id'],v['path'],a,Backend(backend,rec),Path(v['cache_dir']),anchor_config_path=WORK/'anchors.json')
        finally: tracked.select_target=original
        meta['target_selection'].update(rule='original ROI + single-person reacquisition',anchor_x_limit=.95,gap_frames=gap,recovery_events=len(rec.events))
        pre.save_json(pre.cache_paths(v)[1],meta); pre.save_json(WORK/'recovery_events'/f'{sid}.json',rec.events)
        if pre.check_cache(v,mh,verify_anchor=a)!='complete': print('WARNING incomplete',sid)

def audit(rows,mh,anchors,preview):
    from extract_video_kp_cache import KP_DUMP_DTYPE
    report=[]; flags=[]; reps=[]
    for v in rows:
        sid=v['sample_id']; a=anchors.get(sid); state=pre.check_cache(v,mh,verify_anchor=a)
        if state!='complete': flags.append(dict(sample_id=sid,reason=state)); continue
        paths=pre.cache_paths(v); data=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r'); n=len(data)
        with np.load(paths[2],allow_pickle=False) as z: status=z['status']; boxes=z['selected_box']; persons=z['person_count']
        sel=status==0; kp=data['kp']; valid=np.isfinite(kp[:,:,:2]).all(axis=2)&(kp[:,:,2]>=.3)
        gt=np.zeros(n,bool)
        for r in v['annotations']:
            s,e=int(r['first_frame']),int(r['last_frame']); gt[s:e+1]=True
            reps.append(dict(sample_id=sid,orientation=r['person_orientation'],repetition_number=r['repetition_number'],correctness=r['correctness'],selected_pct=round(100*sel[s:e+1].mean(),3),valid_kp_pct=round(100*valid[s:e+1].mean(),3)))
        center=(boxes[:,:2]+boxes[:,2:])/2; scale=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
        jumps=np.linalg.norm(np.diff(center,axis=0),axis=1)/scale[:-1]; adjacent=sel[1:]&sel[:-1]; ids=np.flatnonzero(adjacent&(jumps>=.25))
        edges=np.flatnonzero(np.diff(np.r_[False,~sel,False])); gaps=edges[1::2]-edges[::2]
        focus=gt if v['label_scope']=='repetition' else np.ones(n,bool)
        pct=lambda x:round(float(100*x.mean()),3) if x.size else None
        row=dict(sample_id=sid,dataset=v['dataset'],exercise=v['exercise'],person_id=v['person_id'],label_scope=v['label_scope'],frames=n,selected_pct=pct(sel),valid_kp_pct=pct(valid),gt_selected_pct=pct(sel[gt]),gt_valid_kp_pct=pct(valid[gt]),multi_person_frames=int((persons>1).sum()),no_detection=int((status==1).sum()),no_target=int((status==2).sum()),longest_failure=int(gaps.max()) if len(gaps) else 0,max_box_jump_ratio=round(float(jumps[adjacent].max()),3) if adjacent.any() else None)
        report.append(row); reasons=[]
        if not focus.any(): reasons.append('no_valid_gt')
        elif sel[focus].mean()<.99: reasons.append('selection_below_99pct')
        if focus.any() and valid[focus].mean()<.75: reasons.append('valid_kp_below_75pct')
        if (persons>1).any(): reasons.append('multiple_detections')
        if len(ids): reasons.append('large_box_jump')
        if reasons: flags.append(dict(sample_id=sid,reason=','.join(reasons),large_jump_pairs=[[int(i),int(i+1)] for i in ids],**{k:row[k] for k in ('selected_pct','gt_selected_pct','valid_kp_pct','gt_valid_kp_pct')}))
        print(sid,row['selected_pct'],row['gt_selected_pct'],','.join(reasons),flush=True)
        if preview or reasons:
            sample=set(probes(v))
            for i in sorted(ids,key=lambda j:jumps[j],reverse=True)[:2]: sample.update([int(i),int(i+1)])
            if len(gaps): j=int(np.argmax(gaps)); sample.add(int(edges[2*j]+gaps[j]//2))
            multi=np.flatnonzero(persons>1)
            if len(multi): sample.add(int(multi[len(multi)//2]))
            cap=cv2.VideoCapture(v['path']); images=[]
            try:
                for i in sorted(sample):
                    try: frame=pre.read_frame(cap,i)
                    except RuntimeError: continue
                    if sel[i]:
                        x1,y1,x2,y2=boxes[i].astype(int); cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),3)
                        for x,y,c in kp[i]:
                            if c>=.3 and np.isfinite([x,y]).all(): cv2.circle(frame,(int(x),int(y)),4,(0,255,255),-1)
                    cv2.putText(frame,f'{sid} f={i} selected={int(sel[i])}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,255),2)
                    images.append(cv2.resize(frame,(max(1,round(frame.shape[1]*300/frame.shape[0])),300)))
            finally: cap.release()
            if images: pre.image_write(WORK/'qa'/f'{sid}.jpg',np.vstack(images))
    def write_csv(path,records):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='',encoding='utf-8') as f:
            if records: w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    # Scope-specific outputs prevent a pilot from overwriting a full report.
    tag=__import__('hashlib').sha256('\n'.join(v['sample_id'] for v in rows).encode()).hexdigest()[:10]
    out=WORK/'reports'/tag
    write_csv(out/'audit.csv',report); write_csv(out/'repetition_quality.csv',reps); pre.save_json(out/'review_flags.json',flags)
    print(f'Complete {len(report)}/{len(rows)}; flags {len(flags)}; reports {out}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['manifest','auto','review','extract','audit'])
    ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--weights',type=Path,default=WEIGHTS)
    ap.add_argument('--dataset',choices=['rehab26','mv7']); ap.add_argument('--exercise',action='append'); ap.add_argument('--sample',action='append'); ap.add_argument('--pilot',action='store_true'); ap.add_argument('--no-open',action='store_true')
    args=ap.parse_args()
    if args.mode=='manifest': manifest(args.root,args.weights); return
    from extract_video_kp_cache import UltralyticsPoseBackend,sha256_file
    m=pre.load_json(WORK/'manifest.json'); mh=m['weights_sha256']
    if sha256_file(m['weights'])!=mh: raise ValueError('Weights changed')
    rows=m['videos']
    if args.pilot:
        # Three REHAB26 squat views; six MV7 squat/pushup combinations of view and quality.
        rows=[v for v in rows if (v['dataset']=='rehab26' and v['recording_id']=='ex6_Honza') or (v['dataset']=='mv7' and v['person_id']=='subject_001' and v['exercise'] in {'squat','push_up'})]
    rows=[v for v in rows if (not args.dataset or v['dataset']==args.dataset) and (not args.exercise or v['exercise'] in args.exercise) and (not args.sample or v['sample_id'] in args.sample)]
    if not rows: ap.error('No matching videos')
    if args.sample and set(args.sample)-{v['sample_id'] for v in rows}: ap.error('Unknown sample in selection')
    p=WORK/'anchors.json'; anchors=pre.load_json(p) if p.exists() else {}
    if args.mode=='audit': audit(rows,mh,anchors,args.pilot); return
    backend=UltralyticsPoseBackend(m['weights'],device='0',imgsz=640,conf=.25)
    if args.mode in ('auto','review'): auto_review(rows,backend,anchors,args.mode=='review',args.no_open)
    else: extract(rows,backend,anchors,mh)

if __name__=='__main__': main()
