#!/usr/bin/env python3
"""Compare posture models on the same frozen AI2 + exit=2.5 squat replay.
Run at pose_training_offline root. Only trusted local model artifacts.
No changes to training, cache, detector, counter or existing results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from eval_ai_exercise_stream import TARGETS, load_ai, replay, write_csv, sha, baseline
from eval_session_policy import ExitHysteresisDetector

VARIANTS = ("A", "B", "C", "D", "D_w50", "D_w75")
FIXED = ("gt_reps", "detected_matched", "missed", "extra_detections",
         "squat_sessions", "gt_frames", "gt_raw_squat_frames",
         "gt_confirmed_active_frames", "gt_raw_unavailable_frames",
         "unannotated_frames", "unannotated_active_frames", "first_confirm_frame")
BOUNDARY_KEYS = ("gt_rep", "gt_start", "gt_end", "gt_label", "matched",
                 "det_start", "det_end", "win_start", "win_end",
                 "session_id", "session_confirm_frame")

def boundary_signature(rows):
    return [tuple(r.get(k) for k in BOUNDARY_KEYS)
            for r in sorted(rows, key=lambda r: int(r["gt_rep"]))]

def event_signature(path):
    keys = ("start_frame", "end_frame", "detected_start_frame",
            "detected_end_frame", "session_id", "count")
    return [tuple(r.get(k) for k in keys)
            for r in map(json.loads, path.read_text().splitlines())]

def main():
    import eval_offline_stream as E
    from squat_classification import classify
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--video", action="append", choices=list(TARGETS))
    ap.add_argument("--variant", action="append", choices=VARIANTS)
    ap.add_argument("--compare-dir", type=Path, default=Path("results/squat_multiview/compare"))
    ap.add_argument("--weighted-dir", type=Path, default=Path("results/squat_multiview/weighted"))
    ap.add_argument("--policy-dir", type=Path, default=Path("results/session_policy/exit25"))
    args = ap.parse_args()
    vids = list(dict.fromkeys(args.video or TARGETS))
    variants = ["A"] + [v for v in dict.fromkeys(args.variant or VARIANTS) if v != "A"]
    old = json.loads((args.policy_dir/"summary.json").read_text())
    expected_policy = dict(enter_sec=1.5, exit_to_other_sec=2.5,
                           target_to_target_sec=1.5, window_sec=2, replay_sec=6)
    if old["policy"] != expected_policy:
        raise ValueError("Expected fixed exit25 policy")
    old_rows = pd.read_csv(args.policy_dir/"squat_per_video.csv").set_index("video_id")
    if not old_rows.index.is_unique or set(old_rows.index) != set(TARGETS):
        raise ValueError("Baseline video coverage mismatch")
    build = json.loads(Path("results/squat_multiview/build/manifest.json").read_text())
    names = build["feature_names"]
    for root in (args.compare_dir, args.weighted_dir):
        if json.loads((root/"summary.json").read_text())["feature_sha256"] != build["features_sha256"]:
            raise ValueError("Feature provenance mismatch")
    model_paths = {v: (args.weighted_dir if v.startswith("D_w") else args.compare_dir)/"models"/v
                   for v in variants}
    models = {v:E.load_models(model_paths[v], vids, None) for v in variants}
    ai = {vid:load_ai(Path("models/exercise_lopo"),2,vid) for vid in vids}
    for v in variants:
        for vid in vids:
            art = models[v][vid]
            g = "rehab24:"+TARGETS[vid]
            if (art.get("held_out_group") != g or art.get("held_out_video") != vid
                or int(art.get("held_out_person_id",-1)) != int(TARGETS[vid])
                or g in art["training_groups"] or list(art["feature_names"]) != names
                or art.get("features_sha256") != build["features_sha256"]
                or art.get("feature_variant") != "PlusAsymmetry180"
                or art.get("crop_fraction") != .10):
                raise ValueError(f"Incompatible posture artifact {v}/{vid}")
    root=args.out_dir.resolve()
    root.mkdir(parents=True,exist_ok=False)
    rows=[]; extras=[]; records=[]; checks=[]
    with threadpool_limits(limits=1):
        for vid in vids:
            data,meta=E.load_cache(Path("data/cache/kp_full/rehab24_camera17_yolov8m")/f"{vid}.kp.bin")
            fps=float(meta.get("assumed_fps") or meta.get("video",{}).get("fps") or 30)
            gt=E.load_gt(Path.home()/"capstone2/pose_training/REHAB24-6/Segmentation.csv",vid)
            if (not meta.get("complete") or not len(data) or not gt
                or not np.array_equal(data["frame_id"],np.arange(len(data)))
                or any(str(g["person_id"])!=TARGETS[vid] or not 0<=g["start"]<=g["end"]<len(data) for g in gt)):
                raise ValueError(f"Cache or GT mismatch: {vid}")
            ref_bounds=None; ref_events=None
            for v in variants:
                out=root/v/vid; out.mkdir(parents=True)
                factory=E.make_factory(models[v][vid],fps,"no_overlap",classify,
                                       bottom_depth=.2,bottom_knee=120,pre=16,post=22)
                r,e,s=replay(E,data,fps,factory,ExitHysteresisDetector(ai[vid],fps,2.5),1,out,.30,gt)
                old_s=old_rows.loc[vid]
                for k in FIXED:
                    if s[k]!=old_s[k]:
                        raise RuntimeError(f"{v}/{vid}: {k} changed: {s[k]} != {old_s[k]}")
                bs=boundary_signature(r); es=event_signature(out/"reps.jsonl")
                if v=="A":
                    ref_bounds,ref_events=bs,es
                    for k in ("classified","classification_failed","accuracy_classified",
                              "accuracy_end_to_end","bad_recall_end_to_end"):
                        if not np.isclose(s[k],old_s[k],equal_nan=True):
                            raise RuntimeError(f"A failed reference reproduction: {vid}/{k}")
                elif bs!=ref_bounds or es!=ref_events:
                    raise RuntimeError(f"{v}/{vid}: detected boundaries differ from A")
                checks.append(dict(variant=v,video_id=vid,same_boundaries=True,same_gating_counts=True))
                records.append(dict(condition=v,video_id=vid,**s))
                rows.extend(r); extras.extend(e)
                print(v,vid,"matched",s["detected_matched"],"/",s["gt_reps"],
                      "E2E",s["accuracy_end_to_end"],"Bad recall",s["bad_recall_end_to_end"],flush=True)
    overall={}
    for v in variants:
        sub=[s for s in records if s["condition"]==v]
        z=E.summarize([r for r in rows if r["condition"]==v],
                      [r for r in extras if r["condition"]==v])
        for k in ("squat_sessions","gt_frames","gt_raw_squat_frames","gt_confirmed_active_frames",
                  "gt_raw_unavailable_frames","unannotated_frames","unannotated_active_frames"):
            z[k]=sum(s[k] for s in sub)
        overall[v]=z
    if set(vids)==set(TARGETS):
        for k in ("gt_reps","detected_matched","extra_detections","accuracy_end_to_end","bad_recall_end_to_end"):
            if not np.isclose(overall["A"][k],old["squat"]["candidate"][k]):
                raise RuntimeError(f"A full reference mismatch {k}")
    write_csv(root/"per_video.csv",records)
    write_csv(root/"reps_eval.csv",rows)
    write_csv(root/"extra_detections.csv",extras)
    write_csv(root/"boundary_checks.csv",checks)
    result=dict(overall=overall,videos=vids,variants=variants,policy=expected_policy,
                feature_sha256=build["features_sha256"],
                exercise_feature_sha256=sha(baseline.__file__),
                reference_checks=checks,
                notes="Exploratory REHAB24 Cam17 E2E; fixed AI2 exit25; same detected boundaries across variants; no independent final test.")
    (root/"summary.json").write_text(json.dumps(result,indent=2,default=str))
    for v,z in overall.items():
        print(v,z["detected_matched"],"/",z["gt_reps"],"E2E",z["accuracy_end_to_end"],
              "Bad recall",z["bad_recall_end_to_end"])
    print("Saved:",root)

if __name__=="__main__":
    main()
