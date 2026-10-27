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
