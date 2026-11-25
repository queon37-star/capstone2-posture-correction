#!/usr/bin/env python3
"""Offline integration of the frozen squat and push-up branches. No GPU or cache writes.
Run at pose_training_offline root, beside previous evaluators. Fixed FPS only.
Native single-clip regression + synthetic concatenation (not real transitions).
"""
from __future__ import annotations
import argparse, contextlib, json, math
from collections import deque
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from auto_detect_hailo_bridge import AutoDetectPipeline
from eval_pushup_gated_stream import Branch
from eval_session_policy import ExitHysteresisDetector
from eval_ai_exercise_stream import TARGETS, load_ai, sha, baseline


class UnifiedExercisePipeline(AutoDetectPipeline):
    """Preserve the original squat return contract; expose push_events separately.
    Caller supplies the same selected COCO17 pose to both branches. A new branch
    starts at each confirmed session; never replay across a consumed session.
    Squat's existing lying replay cut is deliberately separate from push-up's.
    """
    def __init__(self, base, fps, **kwargs):
        super().__init__(base, fps, **kwargs)
        if not self.gating: raise ValueError('Unified pipeline requires gating')
        self.push_branch = Branch(fps)
        self.push_recent = deque(maxlen=max(1, round(6*fps)))
        self.push_cut = -1
        self.push_events = []
        self.input_selected = False
        self.last_input = -1

    def _observe(self, m, frame_id):
        super()._observe(m, frame_id)
        if self.missing_run >= self.gap_frames:
            self.push_cut = max(self.push_cut, frame_id)

    def _publish_push(self, f, k, selected, published):
        b = self.push_branch
        n = len(b.events)
        b.feed(f, k, selected, published)
        if len(b.events) != n:
            e = dict(b.events[-1])
            e.update(exercise='pushup', session_id=self.current_session_id,
                     session_count=e['rep'], count=len(self.push_events)+1)
            self.push_events.append(e)

    def _start_session(self, exercise, frame_id, t):
        out = super()._start_session(exercise, frame_id, t)
        if exercise == 'pushup':
            onset = int(round(self.detector.candidate_since*self.fps))
            lo = max(self.push_cut+1, frame_id-round(6*self.fps),
                     onset-round(self.detector.window_sec*self.fps)+1)
            pre = [r for r in self.push_recent if r[0] >= lo]
            if pre and [r[0] for r in pre] != list(range(pre[0][0],frame_id)):
                raise RuntimeError('Noncontiguous push-up replay')
            self.push_branch.start(pre[0][0] if pre else frame_id)
            for f,k,s in pre: self._publish_push(f,k,s,frame_id)
        return out

    def _end_session(self, frame_id, t, reason):
        old = self.active_exercise
        if old == 'pushup':
            self.push_branch.stop()
            if frame_id >= 0: self._cut(frame_id-1)
        if frame_id >= 0:
            self.push_cut = max(self.push_cut, frame_id if reason=='person lost' else frame_id-1)
        return super()._end_session(frame_id,t,reason)

    def push_selected(self, kp, selected, *, frame_id):
        if self.closed or frame_id != self.last_input+1:
            raise ValueError('Expected contiguous open fixed-FPS stream')
        self.last_input = frame_id
        k = np.asarray(kp,dtype=np.float32).copy()
        if k.shape != (17,3): raise ValueError(k.shape)
        if not selected: k[:,:2]=np.nan; k[:,2]=0
        # Existing squat code masks confidence at the original threshold;
        # push-up gets the original confidence values as in its validated replay.
        self.input_selected = bool(selected)
        out = super().push(k,frame_id=frame_id,t=frame_id/self.fps)
        if self.active_exercise == 'pushup':
            self._publish_push(frame_id,k,bool(selected),frame_id)
        self.push_recent.append((frame_id,k,bool(selected)))
        return out


def main():
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    import eval_offline_stream as E
    from squat_classification import classify
    import preprocess_rehab24 as pre
    import eval_pushup_counter_side as side
    import train_exercise_heldout as held
    from extract_video_kp_cache import KP_DUMP_DTYPE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=Path('results/unified_stream'))
    p.add_argument('--mixed-cases',type=int,default=3,choices=range(0,4))
    a=p.parse_args(); root=a.out_dir.resolve()
    if root.exists(): raise FileExistsError(root)
    old=Path('results/session_policy/exit25')
    pg=Path('results/pushup/gated_stream')
    old_s=pd.read_csv(old/'squat_per_video.csv').set_index('video_id')
    old_n=pd.read_csv(old/'other_per_video.csv').set_index('sample_id')
    old_p=pd.read_csv(pg/'per_video.csv').query("condition=='gate_replay'").set_index('sample_id')
    old_pe=pd.read_csv(pg/'rep_events.csv').query("condition=='gate_replay'")
    if (len(old_s),len(old_n),len(old_p),len(old_pe))!=(9,18,23,258): raise ValueError('Baseline scope changed')
    posture=E.load_models(Path('models/lopo_heldout/PlusAsymmetry180_crop10'),list(TARGETS),None)
    ai={vid:load_ai(Path('models/exercise_lopo'),2,vid) for vid in TARGETS}
    inv,_=side.inventory(); folds=pd.read_csv(pg/'folds.csv').set_index('group')
    if set(inv.sample_id)!=set(old_p.index) or set(inv.group)!=set(folds.index): raise ValueError('Push-up scope mismatch')
    pa={}
    for g,r in folds.iterrows():
        f=pg/'models'/r.model
        if sha(f)!=r.model_sha256: raise ValueError('Model hash '+g)
        art=joblib.load(f); m=art['metadata']
        if m['heldout_group']!=g or g in m['training_groups'] or m['feature_source_sha256']!=sha(baseline.__file__) or set(art['pipeline'].classes_)!=set(baseline.LABELS): raise ValueError('Model metadata '+g)
        pa[g]=art
    rm=json.loads(Path('data/preprocess/rehab24/batch_manifest.json').read_text())
    mm=json.loads((side.WORK/'manifest.json').read_text())
    if rm['weights_sha256']!=mm['weights_sha256']: raise ValueError('Pose weights mismatch')
    rv={v['sample_id']:v for v in rm['videos']}; mv={v['sample_id']:v for v in mm['videos']}
    people={pid:vid for vid,pid in TARGETS.items()}
    segfile=Path.home()/'capstone2/pose_training/REHAB24-6/Segmentation.csv'
    def cache(v,m,mask):
        if pre.check_cache(v,m['weights_sha256'])!='complete': raise ValueError('Cache '+v['sample_id'])
        paths=pre.cache_paths(v); d=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r')
        with np.load(paths[2],allow_pickle=False) as z: status=z['status']; boxes=z['selected_box']
        if len(d)!=int(v['frames']) or len(status)!=len(d) or len(boxes)!=len(d) or not np.array_equal(d['frame_id'],np.arange(len(d))): raise ValueError('Frames '+v['sample_id'])
        sel=(status==0)&(d['pose_valid']!=0); k=np.array(d['kp'],copy=True)
        if mask: k=np.stack([E.mask_conf(x,.30) for x in k])
        k[~sel,:,:2]=np.nan; k[~sel,:,2]=0
        return k,sel,float(v['fps']),status,boxes
    def squat(vid):
        d,m=E.load_cache(Path('data/cache/kp_full/rehab24_camera17_yolov8m')/f'{vid}.kp.bin')
        fps=float(m.get('assumed_fps') or m.get('video',{}).get('fps') or 30)
        gt=E.load_gt(segfile,vid)
        if not m.get('complete') or not gt or not np.array_equal(d['frame_id'],np.arange(len(d))) or any(str(r['person_id'])!=TARGETS[vid] or not 0<=r['start']<=r['end']<len(d) for r in gt): raise ValueError(vid)
        k=np.stack([E.mask_conf(x,.30) for x in d['kp']]); sel=d['pose_valid']!=0
        k[~sel,:,:2]=np.nan;k[~sel,:,2]=0
        return k,sel,fps,gt
    def mask_gt(n,gt):
        m=np.zeros(n,bool)
        for r in gt: m[int(r['start']):int(r['end'])+1]=True
        return m
    def other_gt(v,status,boxes):
        n=len(status); sel=status==0; ctr=(boxes[:,:2]+boxes[:,2:])/2
        scale=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
        jump=sel[:-1]&sel[1:]&(np.linalg.norm(np.diff(ctr,axis=0),axis=1)/scale[:-1]>=.25)
        gt=[]
        for r in v['annotations']:
            st,ed=int(r['first_frame']),int(r['last_frame'])
            if not 0<=st<=ed<n: raise ValueError('Other GT range')
            if sel[st:ed+1].mean()>=.99 and not jump[st:ed].any(): gt.append(dict(start=st,end=ed))
        return gt
    root.mkdir(parents=True); (root/'logs').mkdir()
    rec=[]; all_events=[]; trans=[]; checks=[]; sqrows=[]; sqextras=[]; mixrows=[]
    def run(sid,k,sel,fps,art,vid):
        d=root/'logs'/sid;d.mkdir()
        factory=E.make_factory(posture[vid],fps,'no_overlap',classify,bottom_depth=.2,bottom_knee=120,pre=16,post=22)
        es=[]; active=[]
        with (d/'stream.log').open('w') as log,contextlib.redirect_stdout(log):
            pipe=UnifiedExercisePipeline(factory(),fps,events_path=d/'events.jsonl',kp_dump_path='',make_base=factory,gating=1,detector=ExitHysteresisDetector(art,fps,2.5),replay_sec=6.,lying_frames=5,gap_frames=max(10,int(fps*.5)))
            pipe.lost_sec=3.
            for i,(x,s) in enumerate(zip(k,sel)):
                es+=pipe.push_selected(x,bool(s),frame_id=i);active.append(pipe.active_exercise)
            es+=pipe.flush()
        ps=pipe.push_events
        log=[json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()]
        for x in log: trans.append(dict(stream=sid,**x))
        for ex,events in [('squat',es),('pushup',ps)]:
            for i,e in enumerate(events,1):
                if int(e['count'])!=i: raise RuntimeError('Noncumulative count '+sid+ex)
                st,ed=bounds(e,ex)
                if not 0<=st<=ed<len(k): raise RuntimeError('Event bounds '+sid+ex)
                all_events.append({**e,'stream':sid,'exercise':ex})
        return es,ps,np.array(active,object),log
    def bounds(e,ex):
        return (int(e['detected_start_frame']),int(e['detected_end_frame'])) if ex=='squat' else (int(e['start_frame']),int(e['end_frame']))
    def overlaps(events,ex,m):
        return sum(bool(m[st:ed+1].any()) for st,ed in (bounds(e,ex) for e in events))
    def emit_row(sid,kind,es,ps,act,mask,**extra):
        rec.append(dict(stream=sid,kind=kind,gt_frames=int(mask.sum()),squat_active_frames=int(((act=='squat')&mask).sum()),pushup_active_frames=int(((act=='pushup')&mask).sum()),squat_count=len(es),pushup_count=len(ps),squat_overlap=overlaps(es,'squat',mask),pushup_overlap=overlaps(ps,'pushup',mask),**extra))
    with threadpool_limits(limits=1):
        for vid in TARGETS:
            k,sel,fps,gt=squat(vid);es,ps,act,log=run(vid,k,sel,fps,ai[vid],vid)
            confirms={e['session_id']:e['frame_id'] for e in log if e['type']=='session_start' and e['exercise']=='squat'}
            rows,extras,excluded=E.evaluate_video(vid,gt,es,lo=0,hi=len(k)-1,min_iou=.2,condition='unified',session_confirms=confirms)
            if excluded: raise RuntimeError('Excluded squat GT')
            s=E.summarize(rows,extras); sqrows+=rows;sqextras+=extras
            keys=['detected_matched','extra_detections','classified','accuracy_end_to_end','bad_recall_end_to_end']
            ok=all(np.isclose(s[x],old_s.loc[vid,x],equal_nan=True) for x in keys)
            emit_row(vid,'squat',es,ps,act,mask_gt(len(k),gt),**s)
            checks.append(dict(stream=vid,kind='squat',same_reference_metrics=ok))
            print(vid,'squat',s['detected_matched'],'pushup false',len(ps),'reference',ok,flush=True)
        for sid,r in old_n.iterrows():
            v=rv[sid];vid=people[str(v['person_id'])]; k,sel,fps,status,boxes=cache(v,rm,True);gt=other_gt(v,status,boxes);mask=mask_gt(len(k),gt)
            if int(mask.sum())!=int(r.gt_other_frames): raise ValueError('Other GT changed '+sid)
            es,ps,act,log=run(sid,k,sel,fps,ai[vid],vid)
            emit_row(sid,'other',es,ps,act,mask,gt_reps=len(gt),source_exercise=v['exercise'])
            checks.append(dict(stream=sid,kind='other',same_reference_metrics=overlaps(es,'squat',mask)==int(r.squat_reps_overlapping_gt)))
            print(sid,'false squat/pushup',overlaps(es,'squat',mask),overlaps(ps,'pushup',mask),flush=True)
        # The selected squat posture artifact is diagnostic-only on MV7; no squat GT there.
        for r in inv.itertuples(index=False):
            sid=r.sample_id;k,sel,fps,_,_=cache(mv[sid],mm,False)
            es,ps,act,log=run(sid,k,sel,fps,pa[r.group],next(iter(TARGETS)))
            ref=old_pe[old_pe.sample_id==sid]
            keys=['start_frame','bottom_frame','end_frame','alignment_flag']
            ok=[tuple(e[x] for x in keys) for e in ps]==list(ref[keys].itertuples(index=False,name=None))
            emit_row(sid,'pushup',es,ps,act,np.ones(len(k),bool),manual_count=int(old_p.loc[sid,'manual_count']),absolute_error=abs(len(ps)-int(old_p.loc[sid,'manual_count'])),alignment_warnings=sum(e['alignment_flag']==1 for e in ps),unassessable=sum(e['alignment_flag']==-1 for e in ps))
            checks.append(dict(stream=sid,kind='pushup',same_reference_metrics=ok))
            print(sid,'pushup',len(ps),'false squat',len(es),'reference',ok,flush=True)
        if a.mixed_cases:
            X,df,_=held.load_inputs(Path('results/exercise_baseline'),2)
            picks=inv[inv.split=='new_group'].drop_duplicates('group').head(a.mixed_cases)
            for j,((vid,pid),r) in enumerate(zip(TARGETS.items(),picks.itertuples(index=False))):
                train=~df.group.isin(['rehab24:'+pid,r.group]).to_numpy()
                if set(df.loc[train,'label'])!=set(held.LABELS): raise ValueError('Mixed fold missing class')
                model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',C=1)).fit(X[train],df.loc[train,'label'])
                art={'pipeline':model}; sk,ss,fps,gt=squat(vid)
                sid=next(s for s in old_n.index if str(rv[s]['person_id'])==pid and rv[s]['exercise']=='lunge')
                ok,os,ofps,st,box=cache(rv[sid],rm,True);og=other_gt(rv[sid],st,box)
                pk,ps,pfps,_,_=cache(mv[r.sample_id],mm,False)
                if not (np.isclose(fps,30) and np.isclose(ofps,30) and np.isclose(pfps,15)): raise ValueError('Mixed test requires 30/15 FPS')
                # Artificial 30-FPS stream, repeat each 15-FPS push-up frame twice.
                pk=np.repeat(pk,2,axis=0);ps=np.repeat(ps,2)
                parts=[('squat',sk,ss,mask_gt(len(sk),gt),len(gt)),('other',ok,os,mask_gt(len(ok),og),len(og)),('pushup',pk,ps,np.ones(len(pk),bool),int(old_p.loc[r.sample_id,'manual_count'])),('squat',sk,ss,mask_gt(len(sk),gt),len(gt))]
                for mode in ['direct','gap4s']:
                    arrays=[];sels=[];segments=[];offset=0
                    for i,(kind,k,s,m,gtcount) in enumerate(parts):
                        if i and mode=='gap4s':
                            n=120;g=np.full((n,17,3),np.nan,np.float32);g[:,:,2]=0
                            arrays.append(g);sels.append(np.zeros(n,bool));segments.append(('gap',offset,offset+n,np.ones(n,bool),0));offset+=n
                        arrays.append(k);sels.append(s);segments.append((kind,offset,offset+len(k),m,gtcount));offset+=len(k)
                    stream=f'mixed_{j+1}_{mode}';es,pe,act,log=run(stream,np.concatenate(arrays),np.concatenate(sels),fps,art,vid)
                    for idx,(kind,lo,hi,m,gtcount) in enumerate(segments):
                        # End-frame attribution; report crossing separately, never count as true GT match.
                        item=dict(stream=stream,segment=idx,kind=kind,start=lo,end_exclusive=hi,reference_count=gtcount,gt_frames=int(m.sum()),heldout_rehab24='rehab24:'+pid,heldout_mv7=r.group)
                        for ex,events in [('squat',es),('pushup',pe)]:
                            item[ex+'_active_frames']=int(((act[lo:hi]==ex)&m).sum())
                            item[ex+'_end_count']=sum(lo<=bounds(e,ex)[1]<hi for e in events)
                            item[ex+'_cross_boundary']=sum(lo<=bounds(e,ex)[1]<hi and bounds(e,ex)[0]<lo for e in events)
                        mixrows.append(item)
                    print(stream,'squat/pushup',len(es),len(pe),flush=True)
    pd.DataFrame(rec).to_csv(root/'per_video.csv',index=False)
    pd.DataFrame(checks).to_csv(root/'reference_checks.csv',index=False)
    pd.DataFrame(all_events).to_csv(root/'rep_events.csv',index=False)
    pd.DataFrame(trans).to_csv(root/'transitions.csv',index=False)
    pd.DataFrame(sqrows).to_csv(root/'squat_reps_eval.csv',index=False)
    pd.DataFrame(mixrows).to_csv(root/'mixed_segments.csv',index=False)
    pv=pd.DataFrame(rec);pu=pv[pv.kind=='pushup'];neg=pv[pv.kind=='other'];sq=pv[pv.kind=='squat']
    summary=dict(squat=E.summarize(sqrows,sqextras),pushup=dict(manual_total=int(pu.manual_count.sum()),auto_total=int(pu.pushup_count.sum()),absolute_error_sum=int(pu.absolute_error.sum()),normalized_count_accuracy=float(1-pu.absolute_error.sum()/pu.manual_count.sum()),alignment_warnings=int(pu.alignment_warnings.sum()),unassessable=int(pu.unassessable.sum()),false_squat=int(pu.squat_count.sum())),other=dict(gt_frames=int(neg.gt_frames.sum()),false_squat_overlap=int(neg.squat_overlap.sum()),false_pushup_overlap=int(neg.pushup_overlap.sum())),false_pushup_on_squat=int(sq.pushup_overlap.sum()),reference_checks=pd.DataFrame(checks).groupby('kind').same_reference_metrics.agg(['sum','count']).astype(int).to_dict('index'),mixed_streams=2*a.mixed_cases,notes='Native clips + synthetic 30 FPS concatenations; repeated frames in mixed push-up only; no actual transition GT. Native reference checks report differences without tuning. Push-up feedback is not Good/Bad. Fixed FPS offline only; no hardware deployment.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    with ZipFile(root/'unified_stream_results.zip','w',ZIP_DEFLATED) as z:
        for f in root.iterdir():
            if f.suffix in ('.json','.csv'):z.write(f,f.name)
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=='__main__': main()
