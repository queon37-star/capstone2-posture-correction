# R2130-12 Hailo-8 Pose → 스쿼트 분석 통합 준비

## 현재 상태
- PC 측 공통 분석과 어댑터는 구현됨. 기존 23개 테스트는 사용자의 PC에서 통과.
- 실행 진입점: scripts/run_squat_hailo_pose.py
- 이벤트 연결: scripts/hailo_squat_bridge.py
- **실제 R2130-12에서 이 실행 스크립트는 아직 검증하지 않음.**
- 현재 기능은 JSONL 저장까지만 포함하며 FastAPI POST는 미구현.

## 1. PC/WSL 회귀 테스트

~~~bash
cd ~/capstone2/pose_training
git pull --ff-only
python -m pytest -q tests/
python -m py_compile scripts/hailo_squat_bridge.py scripts/run_squat_hailo_pose.py
~~~

무하드웨어 단위 테스트는 HailoRT/gi 설치가 필요하지 않음. 직접 실행은 실제 장비에서 진행.

## 2. R2130-12 선행 확인

1. 이전에 동작하던 Hailo YOLOv8m-Pose 샘플을 먼저 그대로 실행하여 HEF/후처리/카메라 경로 정상 여부 확인.
2. Hailo examples 저장소와 .env 위치 확인. 정상 실행했던 예제의 import 경로가 새 스크립트와 같은지도 확인.
3. 개발 PC에서 학습한 models/squat_plus_asymmetry180_crop10_all9.joblib을 R2130-12로 복사. Git에는 모델이 없을 수 있음.
4. 장비에서 Python의 gi, hailo, hailo_apps, numpy, joblib, pandas, sklearn, scipy, matplotlib import를 확인. **WSL/CUDA 버전을 담은 requirements-wsl.txt를 보드에 통째로 설치하지 말 것.**
5. 실제 pose callback이 제공하는 프레임 속도 확인. SQUAT_FPS=30은 기본 가정이므로 보드 프레임 속도와 일치해야 함.

Python 모듈 확인:

~~~bash
python3 -c 'import gi, hailo, joblib, numpy, pandas, sklearn; print("Core imports OK")'
python3 -c 'from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp; print("App import OK")'
~~~

## 3. 실행 예시 — 보드에서만

~~~bash
cd /path/to/capstone2-posture-correction

export HAILO_EXAMPLES_ROOT=/path/to/hailo-rpi5-examples  # 실제 설치 경로
export HAILO_ENV_FILE="$HAILO_EXAMPLES_ROOT/.env"       # 실제 .env가 있을 때
export SQUAT_MODEL_PATH=models/squat_plus_asymmetry180_crop10_all9.joblib
export SQUAT_OUTPUT=results/hailo/live_reps.jsonl
export SQUAT_FPS=30

python3 scripts/run_squat_hailo_pose.py
~~~

- SQUAT_MODEL_PATH 및 SQUAT_OUTPUT은 절대경로 또는 프로젝트 루트 기준 상대경로.
- 입력 장치/파일 옵션은 해당 장비에서 **정상 동작한 GStreamerPoseEstimationApp 버전**의 실행 옵션을 따른다. 기본값이 USB 카메라라고 가정하면 안 됨.
- 최초에는 한 사람만 화면에 들어오도록 구성하고, 시작할 때 직립 자세를 충분히 유지해 초기 캘리브레이션 수행.
- Hailo callback으로 들어온 모든 유효 프레임은 frame_id 증가 후 bridge로 전달. 0명 또는 2명 이상 사람 검출 시 해당 프레임은 missing_pose로 처리.
- 완료된 반복 이벤트는 JSONL에 기록. 정상 중단 시 마지막 post-context 버퍼를 flush.
- GStreamer callback에서 재는 callback_fps는 초기 처리량 지표이며, Hailo 장치의 실시간 모델 추론 지연이나 화면→앱 응답 시간을 직접 증명하지 않음.

## 4. 장비에서 반드시 검증할 사항

- Hailo landmark 좌표를 변환한 결과가 실제 **callback pad 영상의 width/height**와 일치하는지 오버레이를 통해 확인. crop/letterbox/rotate 경로가 있으면 추가 좌표 매핑 필요.
- 보드의 실제 FPS와 SQUAT_FPS 설정 일치 여부.
- 한 명만 검출되는지, 17개 관절 confidence가 PC 코드와 호환되는지.
- 스쿼트 검출 시 REP의 count/posture/unknown 및 classification_error 확인.
- 시작 시 직립 자세 캘리브레이션, 종료 후 22프레임 후행 버퍼에 따른 피드백 지연 측정.
- 서버 POST와 앱 수신은 후속 작업으로 분리.

## 구현 파일
- scripts/hailo_pose_adapter.py: Hailo 랜드마크→COCO17
- scripts/hailo_squat_bridge.py: 관절 전달 및 JSONL 저장
- scripts/run_squat_hailo_pose.py: 공식 GStreamer Pose 구조 기반 실행 스크립트
- tests/test_hailo_squat_bridge.py: 외부 장비 없는 통합 브리지 테스트


## R2130-12 첫 하드웨어 검증: 카운팅 전용 모드

확인된 장비: HailoRT 4.20.0, Tappas 3.31.0, `hailo_apps_infra`,
`~/hailo-rpi5-examples/venv_hailo_rpi5_examples`.
현재 보드에는 NumPy 1.24.2가 있지만 joblib/sklearn/pandas가 없으므로
**패키지 설치 없이 먼저 스쿼트 반복 검출만** 검증한다.
2026-10-28 이후 통합 스크립트는 구형 `hailo_apps_infra`를 우선 import한다.
학습된 joblib 모델은 repository의 `models/`에 이미 존재한다.

~~~bash
cd ~/hailo-rpi5-examples
source setup_env.sh
cd ~/capstone2/pose_training
git pull --ff-only
python -m pytest -q tests/    # pytest가 있을 때만; 없어도 장비 추론은 별도로 가능
python -m py_compile scripts/run_squat_hailo_pose.py scripts/squat_stream_pipeline.py

export HAILO_EXAMPLES_ROOT="$HOME/hailo-rpi5-examples"
export SQUAT_COUNT_ONLY=1
export SQUAT_OUTPUT="results/hailo/count_only_reps.jsonl"
export SQUAT_FPS=30  # 실제 callback_fps 확인 후 수정

python scripts/run_squat_hailo_pose.py \
    --input /ABSOLUTE/PATH/TO/single_person_squats.mp4 \
    --hef-path "$HOME/hailo-rpi5-examples/resources/yolov8m_pose.hef"
~~~

- 입력 영상에는 한 사람의 전신만 나타나야 한다. 다중 인원이 나오는
  `resources/example.mp4`는 Pose 콜백 배선 테스트에는 사용할 수 있지만
  카운터 반복 검증용으로는 부적절하다.
- `posture=unknown`과 `classification_status=disabled_count_only`는
  카운팅 전용 모드에서 **예상되는 정상 출력**이다. 모델 품질을 뜻하지 않는다.
- `valid_pose=0/...`이면 카운터가 아니라 Hailo 검출/랜드마크/좌표 문제를
  먼저 조사한다. frame callback FPS는 최종 화면 응답 지연이 아니다.
- `Ctrl+C`로 중단한 뒤 `results/hailo/count_only_reps.jsonl`에서
  count, start/end frame, knee angle, peak hip depth를 확인한다.
- 나중에 모델 추론 의존성과 scikit-learn 모델 호환성 문제가 해결되면
  `unset SQUAT_COUNT_ONLY`로 Good/Bad 모드를 실행한다.


## PM_029 재현 테스트: 영상 1회 재생

보드에서 `PM_029-Camera17-30fps.mp4`의 메타데이터가
30 FPS, 3090프레임, 103초임을 확인했다.
보드에 설치된 `GStreamerApp.on_eos()`는 파일 소스에서 기본적으로
`seek_simple(..., 0)`을 호출하여 자동 반복한다.
그 결과 3573 callback 프레임, 17 이벤트를 얻은 첫 테스트는
1회 재생 결과로 간주할 수 없다.

개발 브랜치의 `SquatPoseApp`은 `SQUAT_SINGLE_PASS=1`일 때만
EOS에서 `shutdown()`을 호출해 반복 재생을 중지한다.
기존 Hailo 설치 파일은 변경하지 않는다.

~~~bash
cd ~/hailo-rpi5-examples && source setup_env.sh
cd ~/capstone2/pose_training && git pull --ff-only

export HAILO_EXAMPLES_ROOT="$HOME/hailo-rpi5-examples"
export SQUAT_COUNT_ONLY=1
export SQUAT_SINGLE_PASS=1
export SQUAT_FPS=30
export SQUAT_OUTPUT="results/hailo/PM_029_single_pass.jsonl"

python scripts/run_squat_hailo_pose.py \
  --input "$PWD/PM_029-Camera17-30fps.mp4" \
  --hef-path "$HAILO_EXAMPLES_ROOT/resources/yolov8m_pose.hef"

wc -l results/hailo/PM_029_single_pass.jsonl
~~~

예상 정상 종료 표시: `EOS: file finished; stopping without rewind.`
이후 `DONE`에서 1회 재생의 callback 프레임 수와 유효 pose 수를 확인한다.
**콜백 프레임 수는 원본 3090 프레임과 같다고 보장하지 않는다.**
실제 영상 프레임 누락이 있으면 원본 프레임 ID와 콜백 frame_id는 서로 다르다.
단일 재생 결과를 얻은 뒤에야 PC 결과(20회)와 비교한다.


## A/B 실험: 영상 오버레이·디스플레이 제거

장비에서 `get_pipeline_string()`을 확인한 결과,
`SOURCE_PIPELINE`의 1280x720 RGB 전처리, `INFERENCE_PIPELINE_WRAPPER`,
`TRACKER_PIPELINE`, `USER_CALLBACK_PIPELINE`, `DISPLAY_PIPELINE` 순으로 연결된다.
`DISPLAY_PIPELINE`에 `hailooverlay`, `videoconvert`,
`fpsdisplaysink`가 포함되어 있다.

2026-10-28 개발 브랜치에 `SQUAT_HEADLESS=1`을 추가했다.
이는 파이프라인 끝의 `DISPLAY_PIPELINE`만
`fakesink name=squat_headless_sink sync=false async=false`로 교체한다.
실제 설치된 `DISPLAY_PIPELINE` 함수를 사용해 원래 문자열을 계산하고
꼬리 일치 여부를 확인하므로 Hailo 예제 패키지는 수정하지 않는다.
Hailo 추론, 사람 추적, Python 콜백, 카운팅은 그대로 유지한다.
화면 출력은 제공하지 않는다.

~~~bash
cd ~/capstone2/pose_training
git pull --ff-only
python -m py_compile scripts/run_squat_hailo_pose.py

export HAILO_EXAMPLES_ROOT="$HOME/hailo-rpi5-examples"
export SQUAT_COUNT_ONLY=1
export SQUAT_SINGLE_PASS=1
export SQUAT_HEADLESS=1
export SQUAT_FPS=30
export SQUAT_OUTPUT="results/hailo/PM_029_headless.jsonl"

python scripts/run_squat_hailo_pose.py \
  --input "$PWD/PM_029-Camera17-30fps.mp4" \
  --hef-path "$HAILO_EXAMPLES_ROOT/resources/yolov8m_pose.hef" \
  --disable-sync

wc -l results/hailo/PM_029_headless.jsonl
~~~

비교 기준: 동일 PM_029 한 번 재생, sync 해제, YOLOv8m-Pose,
1280x720 전처리, 30 FPS 가정, `SQUAT_COUNT_ONLY=1`.
이전 표시 파이프라인 결과: callback 3090프레임, 19 이벤트, 381.7초,
8.1 callback FPS. 처리 시간과 이벤트 수가 어떻게 변하는지 확인한다.
`SQUAT_HEADLESS`만 켜도 동일한 3090프레임이 콜백에
도달해야 비교 가능하다. 실제 카메라의 엔드투엔드 지연은 별도 평가 필요.


## Hailo-8 batch 크기 A/B 실험 (2026-10-28)

설치된 `GStreamerPoseEstimationApp.__init__()`은 `self.batch_size=2`,
`video_width=1280`, `video_height=720`을 지정하고
바로 `create_pipeline()`을 호출한다. `INFERENCE_PIPELINE`은
`hailonet batch-size={batch_size}`로 이 값을 사용한다.

개발 브랜치의 `SquatPoseApp.get_pipeline_string()`에서
`SQUAT_BATCH_SIZE=1|2|4`를 받아 **파이프라인 문자열 생성 전**에
`self.batch_size`를 변경하도록 했다. 옵션 미설정 시 기존 값 2 유지.
이 방법은 Hailo 설치 파일을 변경하지 않는다.

HailoRT CLI 단독 측정 (YOLOv8m-Pose; 같은 HEF):
| Batch | hw_only FPS | streaming FPS | HW latency |
|---|---:|---:|---:|
| 1 | 16.3315 | 16.3264 | 50.765 ms |
| 2 | 25.988 | 25.9876 | 60.9056 ms |
| 4 | 35.729 | 35.729 | 86.1391 ms |

앱 Batch 2 / `SQUAT_HEADLESS=1` / 단일 영상 / `--disable-sync`:
3090/3090 콜백 프레임, 19회 검출, 2692 유효 pose 프레임,
118.7초, callback 26.0 FPS. 동일 영상 기존 표시 파이프라인은
381.7초, callback 8.1 FPS였다.

**다음 실험: 배치 설정 외에는 모두 동일하게 유지하고 Batch 4 검증.**

~~~bash
cd ~/capstone2/pose_training
git pull --ff-only
python -m py_compile scripts/run_squat_hailo_pose.py

export HAILO_EXAMPLES_ROOT="$HOME/hailo-rpi5-examples"
export SQUAT_COUNT_ONLY=1
export SQUAT_SINGLE_PASS=1
export SQUAT_HEADLESS=1
export SQUAT_BATCH_SIZE=4
export SQUAT_FPS=30
export SQUAT_OUTPUT="results/hailo/PM_029_headless_batch4.jsonl"

python scripts/run_squat_hailo_pose.py \
  --input "$PWD/PM_029-Camera17-30fps.mp4" \
  --hef-path "$HAILO_EXAMPLES_ROOT/resources/yolov8m_pose.hef" \
  --disable-sync

wc -l results/hailo/PM_029_headless_batch4.jsonl
~~~

실행 시작 시 출력된 `Hailo Pose batch_size=4` 및 실제
파이프라인 내 `hailonet ... batch-size=4`를 확인한다.
`DONE`에서 3090 프레임, 유효 pose 수, 반복 횟수,
wall time, callback FPS를 확인한다.

배치 증가로 처리량이 높아져도 **실시간 입력→판정 지연**이
낮다는 뜻은 아니다. 실제 카메라 지연과 1차/2차 피드백 지연을
별도로 측정해야 한다. 또한 NPU 단독 35.729 FPS가
앱 전체 35.729 FPS를 보장하지 않는다.
