#!/usr/bin/env python3
"""Compare frozen LOPO LR with v5 rule detector on identical squat streams.
Place in scripts/ in pose_training_offline. No changes to original pipeline or caches.
This is a squat-stream diagnostic, not an idle or mixed-exercise evaluation.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import time
from collections import deque
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import train_exercise_baseline as baseline
from train_exercise_heldout import TARGETS


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class AIExerciseDetector:
    """Fixed-FPS replay adapter. Causal trailing window; same features as training.
    Other is a non-target label, NOT trained idle. Invalid windows retain the
    confirmed label, matching v5's update semantics. No extra motion gate.
    """
    lying_ratio = 1.0

    def __init__(self, artifact, fps, seconds, confirm_sec=1.5):
        self.pipeline = artifact['pipeline']
        self.fps = float(fps)
        self.window_sec = float(seconds)
        self.confirm_sec = float(confirm_sec)
        self.span = round(seconds * self.fps)
        self.ids = np.floor(np.arange(round(seconds*15))*self.fps/15).astype(int)
        if self.fps < 15 or self.ids[-1] >= self.span:
            raise ValueError('Unsupported fixed-FPS sample grid')
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.span)
        self.candidate = self.candidate_since = self.confirmed = self.last_raw = None
        self.last_info = {}
        self.last_t = None

    def update(self, kp, t):
        if self.last_t is not None:
            dt = t - self.last_t
            if not np.isclose(dt, 1/self.fps, rtol=.02, atol=.001):
                raise ValueError(f'Nonuniform replay timestamp: {dt}; fixed FPS adapter only')
        self.last_t = t
        self.buf.append(np.asarray(kp, dtype=np.float32).copy())
        raw = None
        self.last_info = {'frames': len(self.buf)}
        if len(self.buf) == self.span:
            k = np.stack(self.buf)[self.ids]
            x = baseline.features(k)
            if x is not None and np.isfinite(x).all():
                p = self.pipeline.predict_proba(x[None])[0]
                raw = str(self.pipeline.classes_[np.argmax(p)]).lower()
                self.last_info.update({str(c): float(v) for c,v in zip(self.pipeline.classes_,p)})
        self.last_raw = raw
        if raw is None:
            return self.confirmed
        if raw != self.candidate:
            self.candidate, self.candidate_since = raw, t
        elif t - self.candidate_since >= self.confirm_sec and raw != self.confirmed:
            self.confirmed = raw
        return self.confirmed


def load_ai(root, seconds, vid):
    path = root / f'{seconds}s' / f'{vid}.joblib'
    art = joblib.load(path)
    m = art['metadata']
    g = 'rehab24:' + TARGETS[vid]
    if (m['heldout_video_id'] != vid or m['heldout_group'] != g or g in m['training_groups']
        or m['window_seconds'] != seconds or m['sampling_fps'] != 15 or m['feature_dim'] != 156
        or m['feature_source_sha256'] != sha(baseline.__file__)
        or set(art['pipeline'].classes_) != set(baseline.LABELS)):
        raise ValueError(f'Incompatible held-out artifact {path}')
    return art


def replay(E, data, fps, factory, detector, gating, out, mask, gt):
    reps, trace = [], []
    gt_mask = np.zeros(len(data), bool)
    for g in gt:
        gt_mask[g['start']:g['end']+1] = True
    with (out / 'stream.log').open('w') as log, contextlib.redirect_stdout(log):
        p = E.AutoDetectPipeline(factory(), fps, events_path=out/'events.jsonl', kp_dump_path='',
                                  make_base=factory, gating=gating, detector=detector,
                                  replay_sec=6., lying_frames=5, gap_frames=max(10,int(fps*.5)))
        p.lost_sec = 3.
        start = time.perf_counter()
        for i,row in enumerate(data):
            k = E.mask_conf(row['kp'], mask)
            # The cache marks absent targets explicitly; do not fabricate observations.
            if not row['pose_valid']:
                k = np.full((17,3), np.nan, np.float32)
            f,t = int(row['frame_id']), float(row['t'])
            reps += p.push(k, frame_id=f,t=t)
            trace.append(dict(frame_id=f,t=t,gt_squat=int(gt_mask[i]),raw=p.detector.last_raw,
                              confirmed=p.detector.confirmed,active=p.active_exercise,
                              prob_squat=p.detector.last_info.get('Squat'),
                              prob_pushup=p.detector.last_info.get('Pushup'),
                              prob_other=p.detector.last_info.get('Other')))
        reps += p.flush()
        elapsed = time.perf_counter()-start
    events = [json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
    with (out/'trace.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(trace[0])); w.writeheader(); w.writerows(trace)
    (out/'reps.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reps))
    starts = [e for e in events if e['type']=='session_start' and e['exercise']=='squat']
    confirms = {e['session_id']:e['frame_id'] for e in starts if 'session_id' in e} if gating else None
    rows, extras, excluded = E.evaluate_video(out.name, gt, reps, lo=0, hi=len(data)-1,
                                              min_iou=.2, condition=out.parent.name,
                                              session_confirms=confirms)
    assert excluded == 0
    active=np.array([r['active']=='squat' for r in trace]); raw=np.array([r['raw']=='squat' for r in trace]); valid=np.array([r['raw'] is not None for r in trace]);
    stats={**E.summarize(rows,extras), 'squat_sessions':len(starts),
           'first_confirm_frame':starts[0]['frame_id'] if starts else None,
           'first_gt_frame':gt[0]['start'], 'gt_frames':int(gt_mask.sum()),
           'gt_raw_squat_frames':int((raw&gt_mask).sum()), 'gt_confirmed_active_frames':int((active&gt_mask).sum()),
           'gt_raw_unavailable_frames':int((~valid&gt_mask).sum()),
           'unannotated_frames':int((~gt_mask).sum()), 'unannotated_active_frames':int((active&~gt_mask).sum()),
           'replay_seconds':elapsed, 'max_replay_ms':p.max_replay_ms}
    return rows,extras,stats


def write_csv(path, rows):
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows)))); w.writeheader();w.writerows(rows)
    else: path.write_text('')


def main():
    import eval_offline_stream as E
    from exercise_detector import ExerciseDetector
    from squat_classification import classify
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--cache-dir',type=Path,default=Path('data/cache/kp_full/rehab24_camera17_yolov8m'))
    a.add_argument('--segmentation',type=Path,default=Path.home()/'capstone2/pose_training/REHAB24-6/Segmentation.csv')
    a.add_argument('--posture-models',type=Path,default=Path('models/lopo_heldout/PlusAsymmetry180_crop10'))
    a.add_argument('--ai-models',type=Path,default=Path('models/exercise_lopo'))
    a.add_argument('--out-dir',type=Path,required=True)
    a.add_argument('--video',action='append',choices=list(TARGETS))
    args=a.parse_args(); vids=list(dict.fromkeys(args.video or TARGETS))
    conditions=['off','rule2','rule4','ai2','ai4']
    # Preflight before writing any output.
    posture=E.load_models(args.posture_models,vids,None)
    ai={(s,v):load_ai(args.ai_models,s,v) for v in vids for s in (2,4)}
    args.out_dir.mkdir(parents=True,exist_ok=False)
    root=args.out_dir.resolve(); records=[]; all_rows=[]; all_extras=[]
    with threadpool_limits(limits=1):
        for v in vids:
            data,meta=E.load_cache(args.cache_dir/f'{v}.kp.bin')
            fps=float(meta.get('assumed_fps') or meta.get('video',{}).get('fps') or 30)
            gt=E.load_gt(args.segmentation,v)
            if not meta.get('complete') or not len(data) or not gt or int(data['frame_id'][0])!=0:
                raise ValueError(f'Invalid cache/GT: {v}')
            if any(g['person_id']!=int(TARGETS[v]) or not 0<=g['start']<=g['end']<len(data) for g in gt):
                raise ValueError(f'GT identity/range mismatch: {v}')
            if posture[v].get('held_out_person_id',int(TARGETS[v])) != int(TARGETS[v]):
                raise ValueError(f'Posture held-out identity mismatch: {v}')
            # Same factory/settings as prior offline evaluator.
            factory=E.make_factory(posture[v],fps,'no_overlap',classify,bottom_depth=.2,bottom_knee=120,pre=16,post=22)
            for name in conditions:
                seconds=4 if name.endswith('4') else 2
                detector=AIExerciseDetector(ai[seconds,v],fps,seconds) if name.startswith('ai') else ExerciseDetector(window_sec=seconds,confirm_sec=1.5)
                out=root/name/v; out.mkdir(parents=True)
                rows,extras,stats=replay(E,data,fps,factory,detector,name!='off',out,.30,gt)
                all_rows.extend(rows); all_extras.extend(extras);records.append(dict(condition=name,video_id=v,**stats))
                print(v,name,'matched',stats['detected_matched'],'/',stats['gt_reps'],'extra',stats['extra_detections'],'e2e',stats['accuracy_end_to_end'],'sessions',stats['squat_sessions'],flush=True)
    overall={c:E.summarize([r for r in all_rows if r['condition']==c],[r for r in all_extras if r['condition']==c]) for c in conditions}
    for c in conditions:
        sub=[r for r in records if r['condition']==c]
        for k in ('squat_sessions','gt_frames','gt_raw_squat_frames','gt_confirmed_active_frames','gt_raw_unavailable_frames','unannotated_frames','unannotated_active_frames'):
            overall[c][k]=sum(r[k] for r in sub)
    write_csv(root/'per_video.csv',records);write_csv(root/'reps_eval.csv',all_rows);write_csv(root/'extra_detections.csv',all_extras)
    (root/'summary.json').write_text(json.dumps({'overall':overall,'per_video':records,'videos':vids,'feature_sha256':sha(baseline.__file__), 'notes':'Squat-only fixed-FPS replay. Unannotated is unknown, not idle GT. Off detector events are observational; counter is ungated. AI raw label computed every frame after full window; same 1.5s confirmation as v5.'},indent=2,default=str))
    print('\ncondition | matched/GT | extra | e2e acc | bad recall e2e | sessions')
    for c,s in overall.items(): print(c,s['detected_matched'], '/',s['gt_reps'],s['extra_detections'],s['accuracy_end_to_end'],s['bad_recall_end_to_end'],s['squat_sessions'])
    print('Saved:',root)

if __name__=='__main__': main()
