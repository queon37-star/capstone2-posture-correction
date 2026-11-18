융캡2 C922 실시간 영상 + 자동 운동 판별 + 스쿼트 Good/Bad 통합 (시험용 패치)

기존 원본 scripts/run_squat_pose_client_render.py, scripts/run_squat_hailo_pose.py,
scripts/squat_stream_pipeline.py, scripts/hailo_squat_bridge.py 는 수정하지 않습니다.
동일한 Hailo/TrackedPersonSelector 포즈 파이프라인을 한 번만 수행합니다.
기존 웹 화면(보드 8765)과 추가 자동 판별 패널(보드 8766)을 노트북 브라우저에서 한 페이지로 봅니다.

전제:
 - 기존 scripts/run_squat_pose_client_render.py 정상 작동
 - 기존 scripts/auto_detect_hailo_bridge.py 및 scripts/exercise_detector.py 설치
 - Good/Bad 모델 및 해당 venv의 joblib/sklearn 로딩 확인

보드에 배포:
  scp auto_dashboard_patch.zip cd4team@<BOARD_IP>:/home/cd4team/capstone2/pose_training/
  cd ~/capstone2/pose_training
  python -m zipfile -e auto_dashboard_patch.zip scripts
  python -m py_compile scripts/auto_dashboard_shim.py scripts/run_squat_pose_auto_dashboard.py

보드 터미널:
  cd ~/hailo-rpi5-examples && source setup_env.sh
  v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0
  cd ~/capstone2/pose_training
  export HAILO_EXAMPLES_ROOT="$HOME/hailo-rpi5-examples"
  export SQUAT_BATCH_SIZE=4 SQUAT_HEADLESS=1 SQUAT_PERSON_SELECTION=tracked SQUAT_FPS=30
  export SQUAT_COUNT_ONLY=0
  export SQUAT_MODEL_PATH="models/squat_plus_asymmetry180_crop10_all9.joblib"
  export SQUAT_AUTO_TIMEBASE=wall
  export SQUAT_DASHBOARD_PORT=8765 SQUAT_AUTO_DASHBOARD_PORT=8766
  export SQUAT_PREVIEW_FPS=10 SQUAT_PREVIEW_WIDTH=800 SQUAT_JPEG_QUALITY=72
  export SQUAT_OUTPUT="results/hailo/c922_combo_reps.jsonl"
  export SQUAT_AUTO_OUTPUT="results/hailo/c922_combo_events.jsonl"
  python -u scripts/run_squat_pose_auto_dashboard.py --input /dev/video0 --arch hailo8 \
    --hef-path "$HAILO_EXAMPLES_ROOT/resources/yolov8m_pose.hef" \
    2>&1 | tee /tmp/c922_combo.log

노트북 Windows PowerShell (별도 창):
  ssh -N -o ExitOnForwardFailure=yes \
    -L 18765:127.0.0.1:8765 -L 18766:127.0.0.1:8766 cd4team@<BOARD_IP>
  ※ PowerShell에서는 위 줄바꿈 백슬래시 대신 한 줄로 작성
  웹 브라우저: http://127.0.0.1:18766/
  기존 영상만: http://127.0.0.1:18765/

실행 시 한 프로세스가 카메라 소유. 이전 카메라 프로그램은 Ctrl+C로 종료해야 합니다.
SQUAT_COUNT_ONLY=0 이면 기존 Good/Bad 분류를 사용합니다.
SQUAT_AUTO_TIMEBASE=wall 이면 ExerciseDetector가 실제 시간을 사용합니다.
푸시업 카운팅은 아직 없습니다. 자동 판별은 HEURISTIC 기반이므로 오판할 수 있습니다.
원본 dashboard가 iframe 표시를 막는 경우 기존 영상 링크를 새 탭에서 여세요.

기능 테스트 로그:
  grep -E 'AUTO_EVENT|REP |AUTO frame=|DONE:|Traceback|ERROR' /tmp/c922_combo.log | tail -n 50
  tail -n 10 results/hailo/c922_combo_reps.jsonl
  tail -n 20 results/hailo/c922_combo_events.jsonl
  curl -s http://127.0.0.1:8766/status | python -m json.tool

주의: 이 통합은 기존 스쿼트 카운터를 자동 운동 세션에 따라 gating하지 않습니다.
자동 판별 이벤트와 스쿼트 Good/Bad REP 이벤트는 별도 JSONL이며 서버 POST는 수행하지 않습니다.
