#!/usr/bin/env python3
"""Fixed-FPS push-up branch integration diagnostic on the 23 screened MV7 clips.
Reuses frozen counter, exercise LR feature/training settings and exit hysteresis.
Compare ungated, gated without replay, gated with bounded causal pre-roll.
Not a full mixed-exercise or hardware test. No existing files are overwritten.
"""
from __future__ import annotations
import argparse, json, hashlib
from collections import deque
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from eval_pushup_counter_pilot import PushupCounter
from train_pushup_posture import extract_features, quality, FEATURE_NAMES
from probe_pushup_feedback import THRESHOLDS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def feedback(k, selected, fps):
    q, why = quality(k, selected, fps)
    h = o = np.nan
    if not why:
        try:
            x = extract_features(k)
            h = float(x[FEATURE_NAMES.index('hip_alignment_p10')])
            o = float(x[FEATURE_NAMES.index('hip_offset_abs_p90')])
        except ValueError as exc:
            why.append('feature:' + str(exc))
    if not np.isfinite([h, o]).all():
        if not why: why.append('alignment_unavailable')
        flag = -1
    else:
        flag = int(h < THRESHOLDS['hip_angle_p10_deg'] and o > THRESHOLDS['hip_offset_abs_p90_torso'])
    return dict(**q, quality_reason=';'.join(why), hip_alignment_p10=h,
                hip_offset_abs_p90=o, alignment_flag=flag,
                feedback={1:'alignment',0:'no_alignment_warning',-1:'unassessable'}[flag])


class Branch:
    """Independent session-local counter; all published frame IDs are absolute."""
    def __init__(self, fps):
        self.fps = fps
        self.events = []
        self.offset = None
        self.counter = None
        self.sessions = 0
        self.history = deque(maxlen=int(np.ceil(6*fps))+3)

    def start(self, frame):
        self.sessions += 1
        self.offset = int(frame)
        self.counter = PushupCounter(self.fps)
        self.history.clear()

    def stop(self):
        self.counter = None
        self.offset = None
        self.history.clear()

    def feed(self, frame, kp, selected, published_frame):
        if self.counter is None: raise RuntimeError('inactive branch')
        k = np.asarray(kp, dtype=np.float32).copy()
        if not selected:
            k[:, :2] = np.nan; k[:, 2] = 0
        self.history.append((frame, k, bool(selected)))
        event = self.counter.update(k, frame-self.offset, bool(selected))
        if event is None: return
        event = dict(event)
        for key in ('start_frame','bottom_frame','end_frame'):
            event[key] += self.offset
        st, ed = event['start_frame'], event['end_frame']
        hist = [r for r in self.history if st <= r[0] <= ed]
        if [r[0] for r in hist] != list(range(st,ed+1)):
            raise RuntimeError('counter interval missing history')
        q = feedback(np.stack([r[1] for r in hist]),np.array([r[2] for r in hist],bool),self.fps)
        self.events.append(dict(**event, **q, session_id=self.sessions,
                                total_count=len(self.events)+1, published_frame=published_frame,
                                replayed=ed < published_frame))


def replay(kp, selected, fps, detector, measure):
    """Replay at most six seconds, at most one feature window before raw candidate onset.
    Never replay through a previously active target session or a >=0.5s pose gap.
    No labels, manual counts, or future keypoints enter this policy.
    """
    branches={n:Branch(fps) for n in ('ungated','gate_no_replay','gate_replay')}
    branches['ungated'].start(0)
    recent=deque(maxlen=max(1,round(6*fps)))
    active=None; cut_after=-1; seen_at=None; missing_run=0
    trace=[]; transitions=[]
    gap_frames=max(10,int(fps*.5))  # same threshold as existing bridge
    for f, (k,sel) in enumerate(zip(kp,selected)):
        t=f/fps
        sel=bool(sel)
        k=np.asarray(k,dtype=np.float32).copy()
        if not sel: k[:, :2]=np.nan; k[:, 2]=0
        branches['ungated'].feed(f,k,sel,f)
        # Match the existing bridge's person-lost check.
        m=measure(k)
        if m is None: missing_run+=1
        else: missing_run=0; seen_at=t
        if missing_run>=gap_frames: cut_after=max(cut_after,f)
        if seen_at is not None and t-seen_at>=3.:
            if active in ('squat','pushup'):
                transitions.append(dict(frame_id=f,old=active,new=None,reason='person_lost',replay_frames=0))
            branches['gate_no_replay'].stop(); branches['gate_replay'].stop()
            active=None; detector.reset(); seen_at=None; cut_after=max(cut_after,f)
        label=detector.update(k,t)
        if label!=active:
            old=active
            if old in ('squat','pushup'):
                branches['gate_no_replay'].stop(); branches['gate_replay'].stop()
                cut_after=max(cut_after,f-1)
            active=label
            pre=[]
            if label=='pushup':
                # At confirmation, candidate_since is a past raw-label timestamp.
                onset=int(round(detector.candidate_since*fps))
                lo=max(cut_after+1, f-round(6*fps), onset-round(detector.window_sec*fps)+1)
                pre=[r for r in recent if r[0]>=lo]
                if pre and [r[0] for r in pre]!=list(range(pre[0][0],f)):
                    raise RuntimeError('noncontiguous replay buffer')
                branches['gate_no_replay'].start(f)
                branches['gate_replay'].start(pre[0][0] if pre else f)
                for pf,pk,ps in pre: branches['gate_replay'].feed(pf,pk,ps,f)
            transitions.append(dict(frame_id=f,old=old,new=label,reason='label_change',replay_frames=len(pre)))
        if active=='pushup':
            for name in ('gate_no_replay','gate_replay'): branches[name].feed(f,k,sel,f)
        recent.append((f,k,sel))
        trace.append(dict(frame_id=f,t=t,raw=detector.last_raw,confirmed=label,selected=int(sel),
                         **{n+'_count':len(b.events) for n,b in branches.items()}))
    return branches,trace,transitions


def main():
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    import train_exercise_heldout as held
    from eval_session_policy import ExitHysteresisDetector
    from exercise_detector import measure
    import eval_pushup_counter_side as side
    import preprocess_rehab24 as pre
    from extract_video_kp_cache import KP_DUMP_DTYPE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=Path('results/pushup/gated_stream'))
    p.add_argument('--baseline-dir',type=Path,default=Path('results/exercise_baseline'))
    a=p.parse_args()
    if a.out_dir.exists(): raise FileExistsError(a.out_dir)
    X,df,sources=held.load_inputs(a.baseline_dir,2)
    inv,_=side.inventory()
    previous=side.OUT/'evaluation'
    old=pd.read_csv(previous/'per_video.csv').set_index('sample_id')
    old_reps=pd.read_csv(previous/'candidate_reps.csv')
    old_summary=json.loads((previous/'summary.json').read_text())
    if old_summary['counter_source_sha256']!=sha(Path(__file__).with_name('eval_pushup_counter_pilot.py')):
        raise ValueError('Counter source differs from frozen side evaluation')
    if set(old.index)!=set(inv.sample_id) or len(old_reps)!=258: raise ValueError('Side inventory changed')
    m=json.loads((side.WORK/'manifest.json').read_text())
    pol=json.loads((side.WORK/'mv7_exclusion_policy.json').read_text())
    vids={v['sample_id']:v for v in m['videos']}
    excluded=set(pol['excluded_sample_ids']); groups=pol['person_fold_groups']
    # Fit the exact pooled LR for each held-out MV7 group; all linked subjects excluded.
    a.out_dir.mkdir(parents=True,exist_ok=False)
    (a.out_dir/'models').mkdir(); (a.out_dir/'traces').mkdir()
    artifacts={}; folds=[]; rows=[]; events=[]; transitions=[]
    with threadpool_limits(limits=1):
        for group in sorted(inv.group.unique()):
            # Also verify every MV7 baseline row for linked subjects uses this mapping.
            linked={p for p,g in groups.items() if 'mv7:'+g==group}
            member=df.dataset.eq('mv7') & df.person_id.isin(linked)
            if not member.equals(df.group.eq(group)):
                raise ValueError(f'Group mapping mismatch: {group}')
            train=~member.to_numpy()
            if set(df.loc[train,'label'])!=set(held.LABELS): raise ValueError('Missing class')
            pipeline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',C=1))
            pipeline.fit(X[train],df.loc[train,'label'].to_numpy())
            meta=dict(heldout_group=group,heldout_people=sorted(linked),training_groups=sorted(df.loc[train,'group'].unique()),
                      train_windows=int(train.sum()),excluded_windows=int((~train).sum()),window_seconds=2,sampling_fps=15,
                      feature_dim=156,feature_source_sha256=sha(held.baseline.__file__),inputs=sources)
            art=dict(pipeline=pipeline,metadata=meta); artifacts[group]=art
            path=a.out_dir/'models'/('fold_'+str(len(folds)+1)+'.joblib');joblib.dump(art,path,compress=3)
            folds.append(dict(group=group,train_windows=int(train.sum()),excluded_windows=int((~train).sum()),linked_people=';'.join(sorted(linked)),model=path.name,model_sha256=sha(path)))
        for r in inv.itertuples(index=False):
            sid=r.sample_id; v=vids[sid]; fps=float(v['fps']); n=int(v['frames'])
            if sid in excluded or (v['dataset'],v['exercise'],v['camera'])!=('mv7','push_up','side') or 'mv7:'+groups[v['person_id']]!=r.group or int(v['clip_correctness'])!=int(r.clip_label): raise ValueError('metadata '+sid)
            if n!=r.frames or not np.isclose(fps,r.fps) or pre.check_cache(v,m['weights_sha256'])!='complete': raise ValueError('cache '+sid)
            paths=pre.cache_paths(v); raw=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r')
            with np.load(paths[2],allow_pickle=False) as z: status=z['status'];boxes=z['selected_box']
            if len(raw)!=n or len(status)!=n or len(boxes)!=n or not np.array_equal(raw['frame_id'],np.arange(n)): raise ValueError('frames '+sid)
            selected=(status==0)&(raw['pose_valid']!=0)
            centers=(boxes[:,:2]+boxes[:,2:])/2;diag=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
            jump=(status[:-1]==0)&(status[1:]==0)&(np.linalg.norm(np.diff(centers,axis=0),axis=1)/diag[:-1]>=.25)
            if jump.any(): raise ValueError('Box jump in screened cache '+sid)
            b,t,tr=replay(raw['kp'],selected,fps,ExitHysteresisDetector(artifacts[r.group],fps,2.5),measure)
            expected=old_reps[old_reps.sample_id==sid].sort_values('rep')
            signature=lambda e: (int(e['start_frame']),int(e['bottom_frame']),int(e['end_frame']))
            if [signature(e) for e in b['ungated'].events]!=[signature(e) for e in expected.to_dict('records')]: raise RuntimeError('Ungated boundaries changed '+sid)
            pd.DataFrame(t).to_csv(a.out_dir/'traces'/f'{sid}.csv',index=False)
            transitions.extend(dict(sample_id=sid,**x) for x in tr)
            gt=int(old.loc[sid,'manual_count']); confirmed=[x['frame_id'] for x in t if x['confirmed']=='pushup']
            for name,branch in b.items():
                es=branch.events; count=len(es)
                rows.append(dict(sample_id=sid,group=r.group,split=r.split,clip_label=int(r.clip_label),condition=name,frames=n,fps=fps,manual_count=gt,auto_count=count,signed_error=count-gt,absolute_error=abs(count-gt),sessions=branch.sessions,
                                 first_confirm_frame=confirmed[0] if confirmed else None,pushup_active_frames=len(confirmed),squat_active_frames=sum(x['confirmed']=='squat' for x in t),
                                 replayed_reps=sum(e['replayed'] for e in es),alignment_warnings=sum(e['alignment_flag']==1 for e in es),unassessable_reps=sum(e['alignment_flag']==-1 for e in es)))
                events.extend(dict(sample_id=sid,condition=name,clip_label=int(r.clip_label),**e) for e in es)
            print(sid,'manual',gt,*(f'{k}={len(v.events)}' for k,v in b.items()),flush=True)
    pv=pd.DataFrame(rows);pv.to_csv(a.out_dir/'per_video.csv',index=False)
    pd.DataFrame(events).to_csv(a.out_dir/'rep_events.csv',index=False)
    pd.DataFrame(transitions).to_csv(a.out_dir/'transitions.csv',index=False)
    pd.DataFrame(folds).to_csv(a.out_dir/'folds.csv',index=False)
    def metrics(d):
        return dict(videos=len(d),manual_total=int(d.manual_count.sum()),auto_total=int(d.auto_count.sum()),exact_count_videos=int((d.absolute_error==0).sum()),absolute_error_sum=int(d.absolute_error.sum()),mean_absolute_error=float(d.absolute_error.mean()),normalized_count_accuracy=float(1-d.absolute_error.sum()/d.manual_count.sum()),alignment_warnings=int(d.alignment_warnings.sum()),unassessable_reps=int(d.unassessable_reps.sum()),replayed_reps=int(d.replayed_reps.sum()))
    summary=dict(overall={k:metrics(d) for k,d in pv.groupby('condition')},new_groups={k:metrics(d) for k,d in pv[pv.split=='new_group'].groupby('condition')},folds=len(folds),ungated_boundaries_verified=True,
                 policy=dict(window_sec=2,enter_sec=1.5,exit_sec=2.5,replay_max_sec=6,pre_candidate_context_sec=2,person_lost_sec=3),
                 notes='23 selected MV7 side clips only. No mixed-exercise negative test, no squat counter connected, no repeat GT. Clip labels weak. No warning is not Good. Fixed FPS only; provisional replay policy; no independent final test.')
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with ZipFile(a.out_dir/'gated_stream_results.zip','w',ZIP_DEFLATED) as z:
        for f in a.out_dir.iterdir():
            if f.suffix in ('.csv','.json'): z.write(f,f.name)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
