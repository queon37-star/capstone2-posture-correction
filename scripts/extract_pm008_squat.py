import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE = Path("data/raw/REHAB24-6")
VIDEO = BASE / "PM_008-Camera17-30fps.mp4"
LABELS = BASE / "Segmentation.csv"

OUT_DIR = Path("data/processed/REHAB24-6")
PREVIEW_DIR = Path("results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# 1. Squat segmentation 로드
with open(LABELS, newline="") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

reps = [
    r for r in rows
    if r["video_id"] == "PM_008"
    and r["exercise_id"] == "6"
]

assert reps, "PM_008의 Squat 라벨이 없습니다."

rep = reps[0]

rep_num = int(rep["repetition_number"])
start = int(rep["first_frame"])
end = int(rep["last_frame"])
label = int(rep["correctness"])

print("=== Selected Repetition ===")
print("Video:", VIDEO.name)
print("Rep:", rep_num)
print("Frames:", start, "~", end)
print("Label:", "Correct" if label == 1 else "Incorrect")


# 2. YOLOv8m-Pose 로드
model = YOLO("yolov8m-pose.pt")

cap = cv2.VideoCapture(str(VIDEO))
assert cap.isOpened(), "영상 열기 실패"

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
assert 0 <= start <= end < total_frames, "Segmentation 범위 오류"

cap.set(cv2.CAP_PROP_POS_FRAMES, start)

keypoints_sequence = []
detected_frames = 0

preview_frames = {start, (start + end) // 2, end}


# 3. 전체 Repetition 프레임 추론
for frame_idx in range(start, end + 1):

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Frame {frame_idx} 읽기 실패")

    result = model.predict(
        frame,
        device=0,
        imgsz=640,
        conf=0.25,
        verbose=False
    )[0]

    # 사람 미검출 시 결측값 처리
    kp = np.full((17, 3), np.nan, dtype=np.float32)

    if (
        result.boxes is not None
        and result.keypoints is not None
        and len(result.boxes) > 0
    ):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        areas = (
            (boxes[:, 2] - boxes[:, 0])
            * (boxes[:, 3] - boxes[:, 1])
        )

        # 임시 기준: 면적과 신뢰도가 가장 큰 사람
        person_idx = np.argmax(areas * scores)

        kp = (
            result.keypoints.data[person_idx]
            .cpu().numpy()
            .astype(np.float32)
        )

        detected_frames += 1

    keypoints_sequence.append(kp)

    # 시작 / 중간 / 끝 프레임 시각화
    if frame_idx in preview_frames:
        out = PREVIEW_DIR / f"pm008_rep{rep_num}_frame{frame_idx}.jpg"
        cv2.imwrite(str(out), result.plot())
        print("Saved preview:", out)

    if (frame_idx - start + 1) % 30 == 0:
        print(f"Processed {frame_idx - start + 1}/{end - start + 1}")


cap.release()


# 4. 결과 저장
keypoints_sequence = np.stack(keypoints_sequence)

out_file = OUT_DIR / f"PM_008_rep{rep_num}.npz"

np.savez_compressed(
    out_file,
    keypoints=keypoints_sequence,
    frame_indices=np.arange(start, end + 1),
    label=label,
    video_id="PM_008",
    repetition_number=rep_num
)

print("\n=== Extraction Result ===")
print("Keypoints shape:", keypoints_sequence.shape)
print("Person detected frames:", detected_frames)
print("Total frames:", len(keypoints_sequence))
print("Detection rate:", round(
    detected_frames / len(keypoints_sequence) * 100, 2
), "%")
print("Saved:", out_file)
