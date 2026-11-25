#!/usr/bin/env python3
"""Negative-control replay: held-out REHAB24 Other videos, unchanged v5/AI2.
Place alongside eval_ai_exercise_stream.py. No training or cache writes.
GT Other is only annotated, quality-screened repetition intervals; outside is unknown.
"""
import argparse, contextlib, csv, json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import preprocess_rehab24 as pre
import eval_offline_stream as E
from exercise_detector import ExerciseDetector
from squat_classification import classify
from eval_ai_exercise_stream import AIExerciseDetector, load_ai, write_csv, TARGETS


def run_video(v, data, status, boxes, fps, detector, factory, out):
    n=len(data); selected=status==0
    center=(boxes[:,:2]+boxes[:,2:])/2
    scale=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
    jumps=selected[:-1]&selected[1:]&(np.linalg.norm(np.diff(center,axis=0),axis=1)/scale[:-1]>=.25)
    gt=np.zeros(n,bool); intervals=[]
    for r in v['annotations']:
        a,b=int(r['first_frame']),int(r['last_frame'])+1
        if not 0<=a<b<=n: raise ValueError('annotation range')
        if selected[a:b].mean()>=.99 and not jumps[a:b-1].any():
            gt[a:b]=True; intervals.append((a,b))
    if not intervals: raise ValueError(f'No quality GT: {v["sample_id"]}')
    trace=[]; reps=[]
    out.mkdir(parents=True)
    with (out/'stream.log').open('w') as log, contextlib.redirect_stdout(log):
        p=E.AutoDetectPipeline(factory(),fps,events_path=out/'events.jsonl',kp_dump_path='',make_base=factory,gating=1,detector=detector,replay_sec=6.,lying_frames=5,gap_frames=max(10,int(fps*.5)))
        p.lost_sec=3.
        for i,row in enumerate(data):
            k=E.mask_conf(row['kp'],.30)
            if not row['pose_valid']: k=np.full((17,3),np.nan,np.float32)
            reps+=p.push(k,frame_id=int(row['frame_id']),t=float(row['t']))
            trace.append(dict(frame_id=i,t=float(row['t']),gt_other=int(gt[i]),raw=p.detector.last_raw,confirmed=p.detector.confirmed,active=p.active_exercise,prob_squat=p.detector.last_info.get('Squat'),prob_pushup=p.detector.last_info.get('Pushup'),prob_other=p.detector.last_info.get('Other')))
        reps+=p.flush()
    write_csv(out/'trace.csv',trace)
    (out/'reps.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reps))
    ev=[json.loads(l) for l in (out/'events.jsonl').read_text().splitlines()]
    raw=np.array([r['raw'] for r in trace],object); active=np.array([r['active'] for r in trace],object)
    false_reps=0
    for r in reps:
        a,b=int(r['detected_start_frame']),int(r['detected_end_frame'])+1
        if not 0<=a<b<=n: raise ValueError('detected range')
        false_reps+=int(gt[a:b].any())
    stats=dict(gt_other_frames=int(gt.sum()),gt_other_reps=len(intervals),total_frames=n,raw_unavailable=int(((raw==None)&gt).sum()),squat_raw=int(((raw=='squat')&gt).sum()),pushup_raw=int(((raw=='pushup')&gt).sum()),squat_active=int(((active=='squat')&gt).sum()),pushup_active=int(((active=='pushup')&gt).sum()),squat_sessions=sum(e['type']=='session_start' and e['exercise']=='squat' for e in ev),pushup_sessions=sum(e['type']=='session_start' and e['exercise']=='pushup' for e in ev),squat_reps_overlapping_gt=false_reps,squat_reps_whole_video=len(reps),gt_reps_with_any_squat_active=sum(bool((active[a:b]=='squat').any()) for a,b in intervals),gt_reps_with_any_pushup_active=sum(bool((active[a:b]=='pushup').any()) for a,b in intervals))
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--camera',type=int,choices=[17,18],default=17)
    p.add_argument('--exercise',action='append',choices=['table_pushup','lunge'])
    p.add_argument('--person',action='append',choices=list(TARGETS))
    a=p.parse_args()
    m=pre.load_json(Path('data/preprocess/rehab24/batch_manifest.json'))
    people={str(pid):vid for vid,pid in TARGETS.items() if not a.person or vid in a.person}
    exercises=set(a.exercise or ['table_pushup','lunge'])
    vs=[v for v in m['videos'] if str(v['person_id']) in people and int(v['camera'])==a.camera and v['exercise'] in exercises]
    if not vs: raise ValueError('No matching negative-control videos')
    root=a.out_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    records=[]; skipped=[]
    with threadpool_limits(limits=1):
        for v in vs:
            sid=v['sample_id']; vid=people[str(v['person_id'])]
            if pre.check_cache(v,m['weights_sha256'])!='complete': raise ValueError(f'Invalid cache {sid}')
            paths=pre.cache_paths(v); data,meta=E.load_cache(paths[0]); fps=float(v['fps'])
            if len(data)!=v['frames'] or not np.array_equal(data['frame_id'],np.arange(len(data))): raise ValueError(sid)
            with np.load(paths[2],allow_pickle=False) as z: status=z['status']; boxes=z['selected_box']
            if len(status)!=len(data) or len(boxes)!=len(data): raise ValueError(sid)
            # No verified identity in GT -> exclude this video if no usable interval.
            ctr=(boxes[:,:2]+boxes[:,2:])/2; scale=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1); sel=status==0
            jumps=sel[:-1]&sel[1:]&(np.linalg.norm(np.diff(ctr,axis=0),axis=1)/scale[:-1]>=.25)
            if not any(0<=int(r['first_frame'])<=int(r['last_frame'])<len(data) and sel[int(r['first_frame']):int(r['last_frame'])+1].mean()>=.99 and not jumps[int(r['first_frame']):int(r['last_frame'])].any() for r in v['annotations']):
                skipped.append(dict(sample_id=sid,reason='no_quality_gt')); continue
            art=load_ai(Path('models/exercise_lopo'),2,vid)
            posture=E.load_models(Path('models/lopo_heldout/PlusAsymmetry180_crop10'),[vid],None)[vid]
            if str(posture.get('held_out_person_id'))!=str(v['person_id']): raise ValueError('posture identity mismatch')
            factory=E.make_factory(posture,fps,'no_overlap',classify,bottom_depth=.2,bottom_knee=120,pre=16,post=22)
            for name,det in [('rule2',ExerciseDetector(window_sec=2,confirm_sec=1.5)),('rule4',ExerciseDetector(window_sec=4,confirm_sec=1.5)),('ai2',AIExerciseDetector(art,fps,2))]:
                s=run_video(v,data,status,boxes,fps,det,factory,root/name/sid)
                records.append(dict(condition=name,sample_id=sid,person_id=str(v['person_id']),exercise=v['exercise'],camera=a.camera,**s))
                print(sid,name,'GT',s['gt_other_frames'],'false squat active',s['squat_active'],'false pushup active',s['pushup_active'],'squat reps overlap',s['squat_reps_overlapping_gt'],flush=True)
    if not records: raise ValueError('No evaluated videos')
    write_csv(root/'per_video.csv',records)
    agg=[]
    for c in ('rule2','rule4','ai2'):
        for ex in ['all']+sorted(exercises):
            sub=[r for r in records if r['condition']==c and (ex=='all' or r['exercise']==ex)]
            if not sub: continue
            keys=[k for k in sub[0] if k not in ('condition','sample_id','person_id','exercise','camera')]
            s={k:sum(r[k] for r in sub) for k in keys}
            agg.append(dict(condition=c,exercise=ex,videos=len(sub),people=len({r['person_id'] for r in sub}),**s,squat_active_pct=100*s['squat_active']/s['gt_other_frames'],pushup_active_pct=100*s['pushup_active']/s['gt_other_frames']))
    (root/'summary.json').write_text(json.dumps(dict(overall=agg,skipped=skipped,notes='Other GT only on selected annotated repetitions; full-video detections diagnostic; pushup sessions measured, no pushup counter implemented. Frozen same-person LOPO models.'),indent=2))
    print(json.dumps(agg,indent=2)); print('Saved:',root)

if __name__=='__main__': main()
