import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path("data/raw/REHAB24-6")
OUT = Path("data/processed/REHAB24-6/camera17_yolov8m")

OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
KEYPOINT_CONF_THRESHOLD = 0.3

# 1. 라벨 로드
with open(BASE / "Segmentation.csv", newline="") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

squats = [r for r in rows if r["exercise_id"] == "6"]

by_video = defaultdict(list)
for r in squats:
    by_video[r["video_id"]].append(r)

print("Total squat repetitions:", len(squats))
print("Total recordings:", len(by_video))

# 2. YOLO 모델 로드
model = YOLO("yolov8m-pose.pt")

metadata = []

# 3. 영상별 처리
for video_id, reps in sorted(by_video.items()):

    video = BASE / f"{video_id}-Camera17-30fps.mp4"
    assert video.exists(), f"Missing video: {video}"

    print(f"\n=== Processing {video_id} ===", flush=True)

    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened(), f"Cannot open: {video}"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cursor = 0

    reps.sort(key=lambda r: int(r["first_frame"]))

    for rep in reps:

        rep_num = int(rep["repetition_number"])
        person_id = int(rep["person_id"])
        start = int(rep["first_frame"])
        end = int(rep["last_frame"])
        label = int(rep["correctness"])

        assert 0 <= start <= end < total
        assert cursor <= start, "Overlapping or unsorted segments"

        output = OUT / f"{video_id}_rep{rep_num:02d}.npz"

        # 반복 시작 전까지 영상 프레임 건너뛰기
        while cursor < start:
            if not cap.grab():
                raise RuntimeError(f"Cannot skip frame {cursor}")
            cursor += 1

        if output.exists():
            with np.load(output) as saved:
                sequence = saved["keypoints"]

            expected = end - start + 1
            assert sequence.shape == (expected, 17, 3)

            while cursor <= end:
                if not cap.grab():
                    raise RuntimeError(f"Cannot skip frame {cursor}")
                cursor += 1

            print(f"  Rep {rep_num}: existing result reused")

        else:
            sequence = []
            multi_person_frames = 0

            # 반복 구간을 배치 단위로 추론
            while cursor <= end:

                frames = []

                for _ in range(min(BATCH_SIZE, end - cursor + 1)):
                    ok, frame = cap.read()

                    if not ok:
                        raise RuntimeError(
                            f"Frame read failed: {video_id}, {cursor}"
                        )

                    frames.append(frame)
                    cursor += 1

                results = model.predict(
                    frames,
                    device=0,
                    imgsz=640,
                    conf=0.25,
                    verbose=False
                )

                for result in results:

                    kp = np.zeros((17, 3), dtype=np.float32)
                    kp[:, :2] = np.nan

                    if (
                        result.boxes is not None
                        and result.keypoints is not None
                        and len(result.boxes) > 0
                    ):
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()

                        if len(boxes) > 1:
                            multi_person_frames += 1

                        areas = (
                            (boxes[:, 2] - boxes[:, 0])
                            * (boxes[:, 3] - boxes[:, 1])
                        )

                        # 임시 피험자 선택 기준
                        idx = np.argmax(areas * scores)

                        kp = (
                            result.keypoints.data[idx]
                            .cpu().numpy()
                            .astype(np.float32)
                        )

                        # 신뢰도가 낮은 관절 좌표는 결측 처리
                        low_conf = kp[:, 2] < KEYPOINT_CONF_THRESHOLD
                        kp[low_conf, :2] = np.nan

                    sequence.append(kp)

            sequence = np.stack(sequence)

            np.savez_compressed(
                output,
                keypoints=sequence,
                frame_indices=np.arange(start, end + 1),
                label=label,
                person_id=person_id,
                video_id=video_id,
                repetition_number=rep_num,
                camera_id=17,
            )

            if multi_person_frames:
                print(
                    f"  WARNING: {multi_person_frames} frames "
                    "contain multiple detected people"
                )

            print(
                f"  Rep {rep_num:2d}: "
                f"{len(sequence)} frames, "
                f"label={label}"
            )

        # 4. 반복별 품질 및 메타데이터 기록
        valid_ratio = float(
            np.mean(sequence[:, :, 2] >= KEYPOINT_CONF_THRESHOLD)
        )

        detected_frames = int(
            np.sum(np.any(sequence[:, :, 2] > 0, axis=1))
        )

        metadata.append({
            "video_id": video_id,
            "person_id": person_id,
            "repetition_number": rep_num,
            "start_frame": start,
            "end_frame": end,
            "orientation": rep["cam17_orientation"],
            "label": label,
            "frames": len(sequence),
            "detected_frames": detected_frames,
            "valid_keypoint_ratio": round(valid_ratio, 4),
            "file": output.name,
        })

    cap.release()

# 5. 인덱스 파일 저장
index_file = OUT / "index.csv"

with open(index_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(metadata[0].keys())
    )
    writer.writeheader()
    writer.writerows(metadata)

print("\n=== Extraction Completed ===")
print("Total repetitions:", len(metadata))
print("Correct:", sum(m["label"] == 1 for m in metadata))
print("Incorrect:", sum(m["label"] == 0 for m in metadata))
print(
    "Mean valid keypoint ratio:",
    round(np.mean([m["valid_keypoint_ratio"] for m in metadata]), 4)
)
print("Saved:", index_file)
