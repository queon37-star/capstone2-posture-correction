from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BASE = Path("data/raw/REHAB24-6")
TRACKED = Path("data/processed/REHAB24-6/camera17_yolov8m_tracked")
PRED_PATH = Path("results/squat_baseline/lopo_predictions.csv")
INDEX_PATH = TRACKED / "index.csv"
OUT = Path("results/squat_error_analysis")

OUT_SINGLE = OUT / "single"
OUT_COMPARE = OUT / "compare"

OUT_SINGLE.mkdir(parents=True, exist_ok=True)
OUT_COMPARE.mkdir(parents=True, exist_ok=True)

COCO_EDGES = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 6),              # shoulders
    (5, 11), (6, 12),    # torso
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (0, 1), (0, 2),      # face
    (1, 3), (2, 4)       # eyes/ears
]

PRED = pd.read_csv(PRED_PATH)
INDEX = pd.read_csv(INDEX_PATH)

pred_cols = [c for c in PRED.columns if "LogisticRegression" in c and c.startswith("pred")]
if not pred_cols:
    raise RuntimeError("Could not find LogisticRegression prediction column.")
PRED_COL = pred_cols[0]

df = PRED.merge(
    INDEX,
    on=["video_id", "person_id", "repetition_number", "label"],
    how="left"
)

if df["file"].isna().any():
    missing = df[df["file"].isna()][["video_id", "repetition_number"]]
    raise RuntimeError(f"Missing tracked npz file mapping:\n{missing}")

errors = df[df["label"] != df[PRED_COL]].copy()

def label_name(x):
    return "Correct" if int(x) == 1 else "Incorrect"

def error_type(label, pred):
    if int(label) == 1 and int(pred) == 0:
        return "FalsePositive"
    if int(label) == 0 and int(pred) == 1:
        return "FalseNegative"
    return "Unknown"

def draw_pose(frame, kp, title_lines):
    img = frame.copy()

    valid_xy = np.isfinite(kp[:, :2]).all(axis=1) & (kp[:, 2] > 0)

    # bbox from keypoints
    if valid_xy.any():
        xs = kp[valid_xy, 0]
        ys = kp[valid_xy, 1]
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        pad = 25
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1] - 1, x2 + pad)
        y2 = min(img.shape[0] - 1, y2 + pad)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # skeleton
    for a, b in COCO_EDGES:
        if valid_xy[a] and valid_xy[b]:
            pa = tuple(np.round(kp[a, :2]).astype(int))
            pb = tuple(np.round(kp[b, :2]).astype(int))
            cv2.line(img, pa, pb, (0, 255, 255), 3)

    # joints
    for i in range(len(kp)):
        if valid_xy[i]:
            p = tuple(np.round(kp[i, :2]).astype(int))
            cv2.circle(img, p, 4, (255, 0, 255), -1)

    # header box
    header_h = 28 * len(title_lines) + 16
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], header_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    y = 28
    for line in title_lines:
        cv2.putText(
            img, line, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
        )
        y += 28
    return img

def load_rep_data(row):
    npz_path = TRACKED / row["file"]
    data = np.load(npz_path)
    return {
        "keypoints": data["keypoints"],
        "frame_indices": data["frame_indices"],
        "video_id": row["video_id"],
        "person_id": int(row["person_id"]),
        "repetition_number": int(row["repetition_number"]),
        "label": int(row["label"]),
        "pred": int(row[PRED_COL]),
        "file": row["file"],
    }

def get_video_path(video_id):
    path = BASE / f"{video_id}-Camera17-30fps.mp4"
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def read_frame(video_path, frame_id):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")
    return frame

def make_triptych(rep_data, out_path, prefix_text):
    kp_seq = rep_data["keypoints"]
    frame_ids = rep_data["frame_indices"]
    video_path = get_video_path(rep_data["video_id"])

    pos = [0, len(frame_ids) // 2, len(frame_ids) - 1]
    phase_names = ["start", "middle", "end"]
    panels = []

    for idx, phase in zip(pos, phase_names):
        frame = read_frame(video_path, frame_ids[idx])
        kp = kp_seq[idx]

        title_lines = [
            prefix_text,
            f"{rep_data['video_id']} rep {rep_data['repetition_number']} | person {rep_data['person_id']}",
            f"{phase} frame={int(frame_ids[idx])} | true={label_name(rep_data['label'])} | pred={label_name(rep_data['pred'])}",
        ]
        panels.append(draw_pose(frame, kp, title_lines))

    h = min(p.shape[0] for p in panels)
    resized = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    sheet = cv2.hconcat(resized)
    cv2.imwrite(str(out_path), sheet)

def make_comparison_sheet(rep_a, rep_b, out_path):
    # rep_a = error sample, rep_b = reference correct sample
    def make_row(rep_data, row_title):
        kp_seq = rep_data["keypoints"]
        frame_ids = rep_data["frame_indices"]
        video_path = get_video_path(rep_data["video_id"])

        pos = [0, len(frame_ids) // 2, len(frame_ids) - 1]
        phase_names = ["start", "middle", "end"]
        panels = []

        for idx, phase in zip(pos, phase_names):
            frame = read_frame(video_path, frame_ids[idx])
            kp = kp_seq[idx]
            title_lines = [
                row_title,
                f"{rep_data['video_id']} rep {rep_data['repetition_number']} | person {rep_data['person_id']}",
                f"{phase} frame={int(frame_ids[idx])} | true={label_name(rep_data['label'])} | pred={label_name(rep_data['pred'])}",
            ]
            panels.append(draw_pose(frame, kp, title_lines))

        h = min(p.shape[0] for p in panels)
        resized = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
        return cv2.hconcat(resized)

    row1 = make_row(rep_a, "MISCLASSIFIED")
    row2 = make_row(rep_b, "REFERENCE (correctly classified)")
    w = max(row1.shape[1], row2.shape[1])

    def pad_to_width(img, w):
        if img.shape[1] == w:
            return img
        pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
        return cv2.hconcat([img, pad])

    row1 = pad_to_width(row1, w)
    row2 = pad_to_width(row2, w)

    sheet = cv2.vconcat([row1, row2])
    cv2.imwrite(str(out_path), sheet)

summary_rows = []

for _, row in errors.iterrows():
    rep = load_rep_data(row)

    err_type = error_type(rep["label"], rep["pred"])
    stem = f"{rep['video_id']}_rep{rep['repetition_number']:02d}_{err_type}"

    # single visualization
    single_out = OUT_SINGLE / f"{stem}.jpg"
    make_triptych(
        rep,
        single_out,
        prefix_text=err_type
    )

    # find reference repetition:
    # same video, same true label, correctly predicted, nearest repetition number
    candidates = df[
        (df["video_id"] == rep["video_id"]) &
        (df["label"] == rep["label"]) &
        (df[PRED_COL] == df["label"]) &
        (df["repetition_number"] != rep["repetition_number"])
    ].copy()

    ref_rep_num = None
    compare_out = None

    if not candidates.empty:
        candidates["dist"] = (candidates["repetition_number"] - rep["repetition_number"]).abs()
        ref_row = candidates.sort_values(["dist", "repetition_number"]).iloc[0]
        ref = load_rep_data(ref_row)

        ref_rep_num = int(ref["repetition_number"])
        compare_out = OUT_COMPARE / f"{stem}_vs_rep{ref_rep_num:02d}.jpg"
        make_comparison_sheet(rep, ref, compare_out)

    summary_rows.append({
        "video_id": rep["video_id"],
        "person_id": rep["person_id"],
        "repetition_number": rep["repetition_number"],
        "true_label": rep["label"],
        "pred_label": rep["pred"],
        "error_type": err_type,
        "single_image": str(single_out),
        "reference_rep": ref_rep_num,
        "compare_image": str(compare_out) if compare_out else "",
    })

summary = pd.DataFrame(summary_rows)
summary_path = OUT / "misclassified_summary.csv"
summary.to_csv(summary_path, index=False)

print("Prediction column:", PRED_COL)
print("Total errors:", len(summary))
print("Saved summary:", summary_path)
print("Saved single visualizations to:", OUT_SINGLE)
print("Saved comparison visualizations to:", OUT_COMPARE)
