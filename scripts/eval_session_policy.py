#!/usr/bin/env python3
"""Exploratory comparison of asymmetric confirmation on frozen AI2 models.
Place in scripts/ next to the two previous stream evaluators. Run at repo root.
Only target -> Other confirmation time changes; no model or cache writes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from eval_ai_exercise_stream import AIExerciseDetector, load_ai, replay, write_csv, TARGETS, sha, baseline


class ExitHysteresisDetector(AIExerciseDetector):
    def __init__(self, artifact, fps, exit_sec):
        super().__init__(artifact, fps, 2, confirm_sec=1.5)
        self.exit_sec = float(exit_sec)
        if self.exit_sec < self.confirm_sec:
            raise ValueError('exit_sec must be >= confirm_sec')

    def update(self, kp, t):
        old = self.confirmed
        # Keep the original causal features, raw labels, candidate updates,
        # and missing-window handling. Delay only target -> Other transitions.
        super().update(kp, t)
        if (old in ('squat', 'pushup') and self.confirmed == 'other'
                and self.last_raw == 'other' and t - self.candidate_since < self.exit_sec):
            self.confirmed = old
        return self.confirmed


def main():
    import eval_offline_stream as E
    import eval_other_exercise_stream as N
    from squat_classification import classify
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--exit-sec', type=float, default=2.5)
    p.add_argument('--squat-baseline',type=Path,default=Path('results/ai_exercise_stream/full'))
    p.add_argument('--other-baseline',type=Path,default=Path('results/other_exercise_stream/baseline'))
    a=p.parse_args()
    if not np.isfinite(a.exit_sec) or a.exit_sec<1.5: p.error('exit-sec must be finite and >=1.5')
    sb=json.loads((a.squat_baseline/'summary.json').read_text())
    nb=json.loads((a.other_baseline/'summary.json').read_text())
    if set(sb['videos'])!=set(TARGETS) or sb['feature_sha256']!=sha(baseline.__file__):
        raise ValueError('Squat baseline scope or feature source changed')
    old_s={r['video_id']:r for r in sb['per_video'] if r['condition']=='ai2'}
    import csv
    with (a.other_baseline/'per_video.csv').open(newline='') as f:
        old_n={r['sample_id']:r for r in csv.DictReader(f) if r['condition']=='ai2'}
    if len(old_s)!=9 or len(old_n)!=18 or nb.get('skipped'):
        raise ValueError('Expected completed 9-video squat and 18-video Other baseline')
    manifest=N.pre.load_json(Path('data/preprocess/rehab24/batch_manifest.json'))
    by_sid={v['sample_id']:v for v in manifest['videos']}
    if not set(old_n)<=set(by_sid): raise ValueError('Other baseline manifest mismatch')
    people={pid:vid for vid,pid in TARGETS.items()}
    posture=E.load_models(Path('models/lopo_heldout/PlusAsymmetry180_crop10'),list(TARGETS),None)
    models={v:load_ai(Path('models/exercise_lopo'),2,v) for v in TARGETS}
    for v in TARGETS:
        if str(posture[v].get('held_out_person_id'))!=TARGETS[v]: raise ValueError(f'Posture identity: {v}')
    root=a.out_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    name='ai2_exit'+str(a.exit_sec).replace('.','p')
    sr=[];nr=[];rows=[];extras=[]
    def factory(v,fps):
        return E.make_factory(posture[v],fps,'no_overlap',classify,bottom_depth=.2,bottom_knee=120,pre=16,post=22)
    with threadpool_limits(limits=1):
        for v in TARGETS:
            data,meta=E.load_cache(Path('data/cache/kp_full/rehab24_camera17_yolov8m')/f'{v}.kp.bin')
            fps=float(meta.get('assumed_fps') or meta.get('video',{}).get('fps') or 30)
            gt=E.load_gt(Path.home()/'capstone2/pose_training/REHAB24-6/Segmentation.csv',v)
            if not meta.get('complete') or not len(data) or not np.array_equal(data['frame_id'],np.arange(len(data))) or not gt or any(str(g['person_id'])!=TARGETS[v] or not 0<=g['start']<=g['end']<len(data) for g in gt):
                raise ValueError(f'Invalid squat input {v}')
            out=root/'squat'/name/v;out.mkdir(parents=True)
            r,e,s=replay(E,data,fps,factory(v,fps),ExitHysteresisDetector(models[v],fps,a.exit_sec),1,out,.30,gt)
            for k in ('gt_reps','gt_frames','unannotated_frames'):
                if s[k]!=old_s[v][k]: raise ValueError(f'Squat baseline denominator changed {v} {k}')
            rows.extend(r);extras.extend(e);sr.append(dict(condition=name,video_id=v,**s))
            print('Squat',v,'matched',s['detected_matched'],'/',s['gt_reps'],'e2e',s['accuracy_end_to_end'],flush=True)
        for sid,old in old_n.items():
            v=by_sid[sid];vid=people[str(v['person_id'])]
            if int(v['camera'])!=17 or v['exercise']!=old['exercise'] or str(v['person_id'])!=old['person_id'] or N.pre.check_cache(v,manifest['weights_sha256'])!='complete':
                raise ValueError(f'Invalid Other input {sid}')
            paths=N.pre.cache_paths(v);data,meta=E.load_cache(paths[0]);fps=float(v['fps'])
            with np.load(paths[2],allow_pickle=False) as z: status=z['status'];boxes=z['selected_box']
            if len(data)!=v['frames'] or len(status)!=len(data) or len(boxes)!=len(data) or not np.array_equal(data['frame_id'],np.arange(len(data))): raise ValueError(sid)
            s=N.run_video(v,data,status,boxes,fps,ExitHysteresisDetector(models[vid],fps,a.exit_sec),factory(vid,fps),root/'other'/name/sid)
            for k in ('gt_other_frames','gt_other_reps','total_frames'):
                if s[k]!=int(old[k]): raise ValueError(f'Other baseline denominator changed {sid} {k}')
            nr.append(dict(condition=name,sample_id=sid,person_id=str(v['person_id']),exercise=v['exercise'],camera=17,**s))
            print('Other',sid,'false squat active',s['squat_active'],'false reps',s['squat_reps_overlapping_gt'],flush=True)
    sq=E.summarize(rows,extras)
    for k in ('squat_sessions','gt_frames','gt_raw_squat_frames','gt_confirmed_active_frames','gt_raw_unavailable_frames','unannotated_frames','unannotated_active_frames'):
        sq[k]=sum(r[k] for r in sr)
    # Raw predictions are not meant to change with session hysteresis. Person-lost
    # resets depend on pose observations, not active labels, in the original bridge.
    for r in sr:
        old=old_s[r['video_id']]
        for k in ('gt_raw_squat_frames','gt_raw_unavailable_frames'):
            if r[k]!=old[k]: raise ValueError(f'Unexpected raw change {r["video_id"]} {k}')
    for r in nr:
        old=old_n[r['sample_id']]
        for k in ('squat_raw','pushup_raw','raw_unavailable'):
            if r[k]!=int(old[k]): raise ValueError(f'Unexpected raw change {r["sample_id"]} {k}')
    neg={}
    for ex in ('all','lunge','table_pushup'):
        sub=[r for r in nr if ex=='all' or r['exercise']==ex]
        keys=[k for k in sub[0] if k not in ('condition','sample_id','person_id','exercise','camera')]
        s={k:sum(r[k] for r in sub) for k in keys}
        s.update(squat_active_pct=100*s['squat_active']/s['gt_other_frames'],pushup_active_pct=100*s['pushup_active']/s['gt_other_frames'])
        neg[ex]=s
    old_neg={r['exercise']:r for r in nb['overall'] if r['condition']=='ai2'}
    write_csv(root/'squat_per_video.csv',sr);write_csv(root/'other_per_video.csv',nr)
    write_csv(root/'reps_eval.csv',rows);write_csv(root/'extra_detections.csv',extras)
    result=dict(policy=dict(enter_sec=1.5,exit_to_other_sec=a.exit_sec,target_to_target_sec=1.5,window_sec=2,replay_sec=6),squat=dict(baseline=sb['overall']['ai2'],candidate=sq),other=dict(baseline=old_neg,candidate=neg),notes='Exploratory on reused held-out subjects. No Idle or mixed transitions GT; missing pose behavior unchanged. Raw counts checked against baseline.')
    (root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    print('Squat baseline/candidate:',sb['overall']['ai2']['detected_matched'],sq['detected_matched'])
    print('Other baseline/candidate active %:',old_neg['all']['squat_active_pct'],neg['all']['squat_active_pct'])
    print('Other baseline/candidate false reps:',old_neg['all']['squat_reps_overlapping_gt'],neg['all']['squat_reps_overlapping_gt'])
    print('Saved:',root)

if __name__=='__main__': main()
