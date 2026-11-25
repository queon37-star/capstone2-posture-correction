#!/usr/bin/env python3
"""Controlled single-person webcam clips: frozen push-up counter and geometry.
Use filename P01_normal_5.mp4 (last number = manually confirmed completed reps).
Conditions: normal, hip_sag, hip_high, shallow. Condition is intended motion,
not a professionally validated error label. No exercise gating in this test.
Run from pose_training_offline, with this file in scripts/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_WEIGHTS="dbe539ea268db2534390942cfdf206e521f376f19e5415967a57f6a2ddfa3c90"
PATTERN=re.compile(r"(?P<person>[A-Za-z0-9-]+)_(?P<condition>normal|hip_sag|hip_high|shallow)_(?P<count>[0-9]+)",re.I)
NAMES=("hip_alignment_p10","hip_offset_abs_p90","elbow_angle_p10","shoulder_ankle_y_p90_p10")

def flags(h,o,e,s,thresholds):
    a=int(h<thresholds["hip_angle_p10_deg"] and o>thresholds["hip_offset_abs_p90_torso"]) if np.isfinite([h,o]).all() else -1
    d=int(e>thresholds["elbow_angle_p10_deg"] and s<thresholds["shoulder_range_torso"]) if np.isfinite([e,s]).all() else -1
    return a,d

def inventory(root):
    paths=sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in {".mp4",".mov",".avi",".mkv"})
    if not paths: raise ValueError(f"No videos under {root}")
    rows=[]
    for path in paths:
        m=PATTERN.fullmatch(path.stem)
        if not m: raise ValueError(f"Invalid filename: {path.name}; use P01_normal_5.mp4")
        rows.append(dict(sample_id=path.stem,person=m["person"].upper(),condition=m["condition"].lower(),manual_count=int(m["count"]),path=str(path.resolve())))
    if len({x["sample_id"].lower() for x in rows})!=len(rows): raise ValueError("Duplicate sample IDs")
    if len({(x["person"],x["condition"]) for x in rows})!=len(rows): raise ValueError("Use one clip per person and condition")
    return rows

def replay(kp,selected,boxes,fps):
    from eval_pushup_counter_pilot import PushupCounter
    from train_pushup_posture import extract_features,quality,FEATURE_NAMES
    from probe_pushup_feedback import THRESHOLDS
    if len(kp)==0 or len(kp)!=len(selected) or len(boxes)!=len(kp): raise ValueError("Empty/misaligned poses")
    kp=np.asarray(kp,dtype=np.float32).copy()
    selected=np.asarray(selected,dtype=bool)
    kp[~selected,:,:2]=np.nan; kp[~selected,:,2]=0
    boxes=np.asarray(boxes,float)
    centers=(boxes[:,:2]+boxes[:,2:])/2
    diag=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
    jumps=selected[:-1]&selected[1:]&(np.linalg.norm(np.diff(centers,axis=0),axis=1)/diag[:-1]>=.25)
    counter=PushupCounter(fps)
    traces=[]; reps=[]
    for i,k in enumerate(kp):
        event=counter.update(k,i,bool(selected[i]))
        traces.append(dict(frame_id=i,t_sec=i/fps,selected=int(selected[i]),depth_norm=counter.value,filtered_depth=counter.counter.filtered,count=len(counter.counter.events)))
        if event is None: continue
        st,ed=event["start_frame"],event["end_frame"]
        q,why=quality(kp[st:ed+1],selected[st:ed+1],fps)
        if jumps[max(0,st-1):min(len(jumps),ed+1)].any(): why.append("box_jump_touching_rep")
        vals=[np.nan]*4
        if not why:
            try:
                x=extract_features(kp[st:ed+1])
                vals=[float(x[FEATURE_NAMES.index(n)]) for n in NAMES]
            except ValueError as exc:
                why.append("feature:"+str(exc))
        a,d=flags(*vals,THRESHOLDS)
        reps.append(dict(**event,**q,quality_reason=";".join(why),**dict(zip(NAMES,vals)),
                         alignment_flag=a,depth_flag_experimental=d,
                         active_feedback="alignment" if a==1 else "no_alignment_warning" if a==0 else "unassessable"))
    return reps,traces,int(jumps.sum())

def read_poses(path,model,device):
    import cv2
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError(f"Cannot open {path}")
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps<=0: cap.release(); raise ValueError(f"Invalid fps {path}")
    poses=[]; selected=[]; boxes=[]; multiple=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            result=model.predict(frame,imgsz=640,conf=.25,device=device,verbose=False)[0]
            k=np.full((17,3),np.nan,np.float32); k[:,2]=0
            b=np.full(4,np.nan,np.float32)
            n=len(result.boxes) if result.boxes is not None else 0
            use=False
            if n>1: multiple+=1
            if n==1 and result.keypoints is not None:
                data=result.keypoints.data.detach().cpu().numpy()
                if data.shape==(1,17,3):
                    k=data[0].astype(np.float32,copy=True)
                    b=result.boxes.xyxy[0].detach().cpu().numpy()
                    use=bool(np.isfinite(k).any() and np.isfinite(b).all())
            poses.append(k); selected.append(use); boxes.append(b)
    finally:
        cap.release()
    if not poses: raise ValueError(f"No decoded frames: {path}")
    return np.stack(poses),np.asarray(selected,bool),np.stack(boxes),fps,multiple

def main():
    from ultralytics import YOLO
    from probe_pushup_feedback import THRESHOLDS
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path.home()/"capstone2/datasets/Pushup-Controlled")
    ap.add_argument("--weights",type=Path,default=Path.home()/"capstone2/pose_training/yolov8m-pose.pt")
    ap.add_argument("--out-dir",type=Path,default=Path("results/pushup/controlled"))
    ap.add_argument("--device",default="0")
    args=ap.parse_args()
    rows=inventory(args.root)
    weight_hash=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if weight_hash!=EXPECTED_WEIGHTS: raise ValueError("Pose weights differ from baseline")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    model=YOLO(str(args.weights))
    all_reps=[]; per_video=[]
    from zipfile import ZipFile,ZIP_DEFLATED
    for row in rows:
        sid=row["sample_id"]
        kp,selected,boxes,fps,multi=read_poses(row["path"],model,args.device)
        reps,traces,jumps=replay(kp,selected,boxes,fps)
        for e in reps: all_reps.append({**row,**e})
        pd.DataFrame(traces).to_csv(args.out_dir/(sid+"_trace.csv"),index=False)
        a=sum(e["alignment_flag"]>=0 for e in reps); b=sum(e["alignment_flag"]==1 for e in reps)
        d=sum(e["depth_flag_experimental"]>=0 for e in reps); f=sum(e["depth_flag_experimental"]==1 for e in reps)
        item=dict(**row,fps=fps,frames=len(kp),selected_pct=float(100*selected.mean()),
                  multiple_person_frames=multi,box_jumps=jumps,auto_count=len(reps),
                  signed_count_error=len(reps)-row["manual_count"],
                  absolute_count_error=abs(len(reps)-row["manual_count"]),
                  alignment_assessable=a,alignment_flagged=b,depth_assessable=d,depth_flagged=f)
        per_video.append(item)
        print(sid, "count",len(reps),"/",row["manual_count"],"alignment",b,"/",a,"depth(exp)",f,"/",d,flush=True)
    v=pd.DataFrame(per_video)
    v.to_csv(args.out_dir/"per_video.csv",index=False)
    pd.DataFrame(all_reps).to_csv(args.out_dir/"rep_feedback.csv",index=False)
    summaries=[]
    for condition,g in v.groupby("condition"):
        n=int(g.alignment_assessable.sum()); d=int(g.depth_assessable.sum())
        summaries.append(dict(condition=condition,clips=len(g),people=g.person.nunique(),
            manual_count=int(g.manual_count.sum()),auto_count=int(g.auto_count.sum()),
            absolute_count_error=int(g.absolute_count_error.sum()),
            alignment_assessable=n,alignment_flagged=int(g.alignment_flagged.sum()),
            alignment_flag_rate=float(g.alignment_flagged.sum()/n) if n else None,
            depth_assessable=d,depth_flagged=int(g.depth_flagged.sum()),
            depth_flag_rate=float(g.depth_flagged.sum()/d) if d else None))
    summary=dict(conditions=summaries,thresholds=THRESHOLDS,weights_sha256=weight_hash,
        notes="Controlled single-person clips; intended variations are not independent per-rep error GT. Counts manually encoded in filenames; no boundary GT. Frozen counter and feedback; depth logged only. No exercise gating. Pose selection requires exactly one detection per frame, not production tracker.")
    (args.out_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    with ZipFile(args.out_dir/"controlled_results.zip","w",ZIP_DEFLATED) as z:
        for f in sorted(args.out_dir.iterdir()):
            if f.suffix in {".csv",".json"}: z.write(f,f.name)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
