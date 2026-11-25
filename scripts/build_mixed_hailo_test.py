#!/usr/bin/env python3
"""Create two synthetic stored-video streams and score Hailo frame/event outputs.
Run build on WSL from repo root. Run score with only Python on board or WSL.
No training, no oracle target selection, no original video writes.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

FPS=30
W,H=1280,720
GAP=120

def load(p): return json.loads(Path(p).read_text())

def build(out):
    import cv2
    import numpy as np
    root=Path.cwd()
    r=load(root/'data/preprocess/rehab24/batch_manifest.json')
    m=load(root/'data/preprocess/extra_datasets/manifest.json')
    rv={v['sample_id']:v for v in r['videos']}; mv={v['sample_id']:v for v in m['videos']}
    sq=rv['Ex6_Camera17_PM_029']; other=rv['Ex3_Camera17_PM_010']
    def squat(label):
        a=sorted(sq['annotations'],key=lambda x:int(x['first_frame']))
        runs=[]; run=[]
        for x in a:
            if int(x['correctness'])==label: run.append(x)
            else:
                if run: runs.append(run)
                run=[]
        if run: runs.append(run)
        if not runs or max(map(len,runs))<3: raise ValueError('No >=3 consecutive labelled squat reps')
        rr=max(runs,key=len)[:5]
        st=int(rr[0]['first_frame']); ed=int(rr[-1]['last_frame'])+1
        # Up to 0.5 s surrounding context, never include other annotated reps.
        prev=[int(x['last_frame'])+1 for x in a if int(x['last_frame'])<st]
        nxt=[int(x['first_frame']) for x in a if int(x['first_frame'])>=ed]
        lo=max(0,st-15,max(prev,default=0)); hi=min(int(sq['frames']),ed+15,min(nxt,default=int(sq['frames'])))
        return dict(v=sq,lo=lo,hi=hi,exercise='squat',quality='good' if label else 'bad',count=len(rr),annotations=rr,label_scope='rep')
    def push(label):
        v=mv[f'MV7_subject_001_push_up_{label}_side']
        # Previously manually counted complete clips: 18 bad and 12 good.
        expected={'bad':(378,18),'good':(434,12)}[label]
        if int(v['frames'])!=expected[0] or float(v['fps'])!=15: raise ValueError('Push-up clip metadata differs from manual reference')
        return dict(v=v,lo=0,hi=int(v['frames']),exercise='pushup',quality=label,count=expected[1],annotations=[],label_scope='clip')
    # Full Other clip: preserve all original non-target activity and breaks.
    plan=[squat(1),push('bad'),push('good'),dict(v=other,lo=0,hi=int(other['frames']),exercise='other',quality=None,count=0,annotations=other['annotations'],label_scope='non_target'),squat(0)]
    if out.exists(): raise FileExistsError(out)
    for p in plan:
        v=p['v']; path=Path(v['path']); fps=float(v['fps'])
        if not path.is_file() or fps not in (15.,30.) or not 0<=p['lo']<p['hi']<=int(v['frames']): raise ValueError(f'Bad source: {path}')
        print(v['sample_id'],p['quality'],p['lo'],p['hi'],p['count'],flush=True)
    out.mkdir(parents=True)
    for mode,gap in [('direct',0),('gap4s',GAP)]:
        video=out/f'{mode}.mp4'; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))
        if not writer.isOpened(): raise RuntimeError('VideoWriter failed')
        segments=[]; n=0
        try:
            for i,p in enumerate(plan):
                if i and gap:
                    segments.append(dict(segment=f'gap{i}',exercise='gap',quality=None,start=n,end=n+gap,expected_count=0,annotations=[]))
                    for _ in range(gap): writer.write(np.zeros((H,W,3),np.uint8))
                    n+=gap
                v=p['v']; cap=cv2.VideoCapture(str(v['path'])); start=n; mul=int(FPS/float(v['fps']))
                try:
                    if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-float(v['fps']))>.01: raise ValueError('Source FPS mismatch')
                    # Sequential decode from frame 0 avoids seek/keyframe ambiguity.
                    for f in range(p['hi']):
                        ok,img=cap.read()
                        if not ok: raise RuntimeError(f'Decode failed {v["sample_id"]} frame {f}')
                        if f<p['lo']: continue
                        h,w=img.shape[:2]; scale=min(W/w,H/h); nw,nh=round(w*scale),round(h*scale)
                        resized=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
                        canvas=np.zeros((H,W,3),np.uint8); x=(W-nw)//2;y=(H-nh)//2;canvas[y:y+nh,x:x+nw]=resized
                        for _ in range(mul): writer.write(canvas); n+=1
                finally: cap.release()
                ann=[dict(start=start+(int(a['first_frame'])-p['lo'])*mul,end=start+(int(a['last_frame'])+1-p['lo'])*mul, label=int(a['correctness']) if p['exercise']=='squat' else None) for a in p['annotations']]
                segments.append(dict(segment=f'{i+1}_{p["exercise"]}_{p["quality"] or "other"}',exercise=p['exercise'],quality=p['quality'],label_scope=p['label_scope'],sample_id=v['sample_id'],source_start=p['lo'],source_end=p['hi'],source_fps=v['fps'],start=start,end=n,expected_count=p['count'],annotations=ann))
        finally: writer.release()
        cap=cv2.VideoCapture(str(video)); actual=0
        try:
            if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-FPS)>.01: raise RuntimeError('Output FPS invalid')
            while cap.grab(): actual+=1
        finally: cap.release()
        if actual!=n: raise RuntimeError(f'Output decoded {actual}, expected {n}')
        (out/f'{mode}.json').write_text(json.dumps(dict(mode=mode,fps=FPS,frames=n,segments=segments,note='Half-open frame intervals. Synthetic cuts between different people/scenes; push-up labels are clip-level, not rep posture ground truth. 15 FPS frames repeated twice.'),indent=2))
        print(mode,'frames',n,'seconds',n/FPS,'file',video,flush=True)

def score(manifest,run):
    m=load(manifest); s=load(run/'summary.json'); seg=m['segments']; fps=m['fps']
    if not s['fixed_fps_smoke_pass'] or s['frames']!=m['frames'] or s['source_fps']!=fps: raise ValueError('Invalid hardware run')
    tr=[json.loads(l) for l in (run/'frame_trace.jsonl').read_text().splitlines()]; ev=[json.loads(l) for l in (run/'reps.jsonl').read_text().splitlines()]
    if [t['frame_id'] for t in tr]!=list(range(m['frames'])): raise ValueError('Trace mismatch')
    if sum(e['exercise']=='squat' for e in ev)!=s['squat_count'] or sum(e['exercise']=='pushup' for e in ev)!=s['pushup_count']: raise ValueError('Event count mismatch')
    rows=[]; cross=[]
    def bounds(e): return (int(e['detected_start_frame']),int(e['detected_end_frame'])+1) if e['exercise']=='squat' else (int(e['start_frame']),int(e['end_frame'])+1)
    for e in ev:
        a,b=bounds(e)
        if not 0<=a<b<=m['frames']: raise ValueError('Invalid rep bounds')
        if not any(z['start']<=a<b<=z['end'] for z in seg): cross.append(dict(exercise=e['exercise'],count=e['count'],start=a,end=b))
    for z in seg:
        lo,hi=z['start'],z['end']; t=tr[lo:hi]; ex=z['exercise']; e=[e for e in ev if lo<=bounds(e)[1]-1<hi]; counts=collections.Counter(q['exercise'] for q in e)
        active=collections.Counter(q['active'] or 'none' for q in t); confirmed=collections.Counter(q['confirmed'] or 'none' for q in t)
        settled=t[min(len(t),4*fps):]; expected_active=ex if ex in ('squat','pushup') else 'none'
        ids=[q['selected_track_id'] for q in t if q['valid_pose'] and q['selected_track_id'] is not None]
        row=dict(segment=z['segment'],exercise=ex,quality=z['quality'],frames=len(t),valid_pose=sum(q['valid_pose'] for q in t),active=dict(active),confirmed=dict(confirmed),settled_active_match_pct=100*sum((q['active'] or 'none')==expected_active for q in settled)/len(settled) if settled else None,selected_ids=dict(collections.Counter(ids)),track_id_changes=sum(a!=b for a,b in zip(ids,ids[1:])),expected_count=z['expected_count'],squat=counts['squat'],pushup=counts['pushup'],cross_boundary_ending=sum(bounds(q)[0]<lo for q in e),squat_good=sum(q['exercise']=='squat' and q.get('posture')=='good' for q in e),squat_bad=sum(q['exercise']=='squat' and q.get('posture')=='bad' for q in e),squat_unknown=sum(q['exercise']=='squat' and q.get('posture')=='unknown' for q in e),pushup_warnings=sum(q['exercise']=='pushup' and q.get('alignment_flag')==1 for q in e),pushup_unassessable=sum(q['exercise']=='pushup' and q.get('alignment_flag')==-1 for q in e))
        if ex=='squat':
            candidates=[]; pred=[q for q in e if q['exercise']=='squat' and bounds(q)[0]>=lo]; gt=z['annotations']
            for i,q in enumerate(pred):
                a,b=bounds(q)
                for j,g in enumerate(gt):
                    inter=max(0,min(b,g['end'])-max(a,g['start'])); union=max(b,g['end'])-min(a,g['start']); iou=inter/union
                    if iou>=.2: candidates.append((iou,i,j))
            usedp=set();usedg=set(); correct=0
            for _,i,j in sorted(candidates,reverse=True):
                if i in usedp or j in usedg: continue
                usedp.add(i);usedg.add(j);correct+=pred[i].get('posture')==('good' if gt[j]['label']==1 else 'bad')
            row.update(squat_gt_matched=len(usedg),squat_gt_correct=correct,squat_gt_missed=len(gt)-len(usedg),squat_unmatched_predictions=counts['squat']-len(usedp))
        rows.append(row); print(json.dumps(row,ensure_ascii=False))
    report=dict(mode=m['mode'],segments=rows,cross_boundary_events=cross,summary=s,note='Event count assigned to detected end frame, not publication time; squat IoU greedy matching >=0.2; push-up clip-level counts and warning counts only; post-cut 4s active measure is descriptive. Other active none includes missing/undecided, not proof of raw Other classification.')
    (run/'mixed_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print('Saved',run/'mixed_report.json')

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build');b.add_argument('--out',type=Path,default=Path('results/mixed_hailo_inputs'))
    q=sub.add_parser('score');q.add_argument('manifest',type=Path);q.add_argument('run',type=Path)
    a=p.parse_args();build(a.out) if a.mode=='build' else score(a.manifest,a.run)
if __name__=='__main__':main()
