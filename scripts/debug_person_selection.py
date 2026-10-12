from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path("data/raw/REHAB24-6")
DATA = Path("data/processed/REHAB24-6/camera17_yolov8m")
OUT = Path("results/person_selection_debug")
OUT.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8m-pose.pt")

cases = [
    ("PM_022", 16, "jump"),
    ("PM_022", 10, "middle"),
    ("PM_038", 10, "middle"),
]

for video_id, rep, mode in cases:
    path = DATA / f"{video_id}_rep{rep:02d}.npz"

    with np.load(path) as data:
        kp = data["keypoints"]
        frame_ids = data["frame_indices"]

    if mode == "jump":
        hips = kp[:, [11, 12], :2]
        confidence = kp[:, [11, 12], 2]

        valid = (
            np.isfinite(hips).all(axis=(1, 2))
            & (confidence >= 0.3).all(axis=1)
        )

        centers = hips.mean(axis=1)
        jumps = np.linalg.norm(
            np.diff(centers, axis=0), axis=1
        )

        jumps[~(valid[:-1] & valid[1:])] = -1

        idx = int(np.argmax(jumps))
        positions = [idx, idx + 1]

        print(
            f"{video_id} Rep {rep}: "
            f"maximum jump = {jumps[idx]:.1f}px"
        )
    else:
        positions = [len(frame_ids) // 2]

    video = BASE / f"{video_id}-Camera17-30fps.mp4"
    cap = cv2.VideoCapture(str(video))

    assert cap.isOpened(), f"Cannot open {video}"

    for pos in positions:
        frame_id = int(frame_ids[pos])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ok, frame = cap.read()
        assert ok, f"Frame {frame_id} read failed"

        result = model.predict(
            frame,
            device=0,
            imgsz=640,
            conf=0.25,
            verbose=False
        )[0]

        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            areas = (
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            )

            selected = int(np.argmax(areas * scores))

            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box.astype(int)

                chosen = i == selected
                color = (0, 0, 255) if chosen else (0, 255, 0)

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    color, 4
                )

                text = (
                    f"{'SELECTED' if chosen else 'OTHER'} "
                    f"{score:.2f}"
                )

                cv2.putText(
                    frame, text,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2
                )

        out_path = OUT / (
            f"{video_id}_rep{rep:02d}_frame{frame_id}.jpg"
        )

        cv2.imwrite(str(out_path), frame)

        print(
            f"Saved: {out_path} | "
            f"Detected people: {len(result.boxes)}"
        )

    cap.release()

print("\nDebug images saved to:", OUT)
