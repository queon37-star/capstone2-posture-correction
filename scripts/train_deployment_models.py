#!/usr/bin/env python3
"""Train final, all-training-data candidates from the frozen offline settings.
Run from pose_training_offline. Does not overwrite held-out models.
This is model preparation, not hardware deployment or independent evaluation.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import train_exercise_heldout as held
import train_exercise_baseline as baseline
import train_squat_multiview as squat

def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def save_verified(artifact, path, X):
    pipe = artifact["pipeline"]
    before = pipe.predict_proba(X)
    joblib.dump(artifact, path, compress=3)
    after = joblib.load(path)["pipeline"].predict_proba(X)
    np.testing.assert_allclose(before, after, atol=1e-12, rtol=0)
    return {"file": path.name, "sha256": sha(path),
            "reload_max_prob_difference": float(np.max(np.abs(before-after)))}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results/deployment_candidates"))
    args = p.parse_args()
    o = args.out_dir
    if o.exists(): raise FileExistsError(o)
    X, df, sources = held.load_inputs(Path("results/exercise_baseline"), 2)
    b = Path("results/squat_multiview/build")
    m = json.loads((b/"manifest.json").read_text())
    template = Path("models/squat_plus_asymmetry180.joblib")
    for path,key in [(b/"features_crop10.npy","features_sha256"),
                     (b/"repetitions.csv","repetitions_sha256"),
                     (template,"template_sha256")]:
        if sha(path) != m[key]: raise ValueError("Hash mismatch: "+str(path))
    d = pd.read_csv(b/"repetitions.csv", dtype={"person_id":str,"group":str,"rep_id":str})
    x = np.load(b/"features_crop10.npy", allow_pickle=False)
    squat.validate(d,x,m)
    use = squat.masks(d)["A"]
    y = d.loc[use,"label"].to_numpy(int)
    xx = x[use].reshape(-1,180)
    yy = np.repeat(y,4)
    if (len(y), int(y.sum()), len(yy)) != (195,134,780):
        raise ValueError("Unexpected A training scope")
    art = joblib.load(template)
    names = list(art["feature_names"])
    if names != m["feature_names"] or art.get("feature_variant")!="PlusAsymmetry180":
        raise ValueError("Wrong squat feature schema")
    if art["pipeline"].steps[-1][1].C != .1:
        raise ValueError("Unexpected squat LR C")
    versions = dict(python=platform.python_version(), sklearn=sklearn.__version__,
                    numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__)
    with threadpool_limits(limits=1):
        ep = make_pipeline(StandardScaler(),LogisticRegression(
            max_iter=2000,class_weight="balanced",C=1)).fit(X,df.label.to_numpy())
        sp = clone(art["pipeline"]).fit(xx,yy)
    em = dict(schema_version=1, purpose="deployment_candidate",window_seconds=2,
              sampling_fps=15,feature_dim=156,labels=list(held.LABELS),
              probability_classes=ep.classes_.tolist(),training_windows=len(df),
              training_counts=held.counts(df),training_groups=sorted(df.group.unique().tolist()),
              feature_source_sha256=sha(baseline.__file__),inputs=sources,versions=versions)
    sa = dict(pipeline=sp,feature_names=names,feature_groups=["base","asymmetry"],
              feature_variant="PlusAsymmetry180",training_variant="A_all9_crop10",
              crop_fraction=.1,label_map={0:"incorrect",1:"correct"},
              training_person_ids=sorted(map(int,d.loc[use,"person_id"].unique())),
              training_groups=sorted(d.loc[use,"group"].unique().tolist()),
              training_repetitions=len(y),training_unique_repetitions=len(y),
              training_feature_rows=len(yy),features_sha256=m["features_sha256"],
              purpose="deployment_candidate")
    o.mkdir(parents=True,exist_ok=False)
    files = dict(exercise=save_verified({"pipeline":ep,"metadata":em},
                   o/"exercise_lr156_2s.joblib",X),
                 squat=save_verified(sa,o/"squat_lr180_A_crop10.joblib",x[use,0,:]))
    config = dict(exercise=dict(window_sec=2,sampling_fps=15,enter_sec=1.5,
                exit_to_other_sec=2.5,target_to_target_sec=1.5),
                stream=dict(replay_sec=6,person_lost_sec=3,pose_confidence=.30,
                            gap_policy="max(10,int(fps*0.5))",lying_frames=5,
                            validated_fixed_fps=[15,30]),
                squat=dict(bottom_depth=.2,bottom_knee=120,pre_frames=16,post_frames=22,
                           boundary_policy="no_overlap",training_camera=17),
                pushup=dict(alignment_warning_enabled=True,hip_angle_p10_deg=160,
                            hip_offset_abs_p90_torso=.2,depth_warning_enabled=False,
                            good_bad_classifier_enabled=False),
                operational_note="Prototype: allow >=4 seconds without a detected person between exercise types; direct switching remains unvalidated.",
                note="Reference settings only; live adapter must explicitly apply them. Not an executable runtime configuration.")
    manifest = dict(models=files,versions=versions,exercise_training=em,
        squat_training={k:v for k,v in sa.items() if k!="pipeline"},
        configuration=config, evaluation_note="All available selected training groups included. No independent test performance claimed. No Hailo or live timing validation.")
    (o/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (o/"reference_config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False))
    print("Exercise windows:",len(df),"groups:",df.group.nunique(),held.counts(df))
    print("Squat reps:",len(y),"augmented rows:",len(yy),"good/bad:",int(y.sum()),int((y==0).sum()))
    print(json.dumps(files,indent=2))
    print("Saved:",o)

if __name__=="__main__":
    main()
