Hailo-8 automatic exercise detection adapter - initial integration test

This is a minimal-risk integration PROBE, not the final full SessionManager + API integration.
No edits to scripts/run_squat_hailo_pose.py or scripts/hailo_squat_bridge.py.

Deploy from Windows:
  scp auto_detect_hailo_patch.zip cd4team@<BOARD_IP>:/home/cd4team/capstone2/pose_training/
On board:
  cd ~/capstone2/pose_training
  python -m zipfile -e auto_detect_hailo_patch.zip scripts
  python scripts/install_hailo_auto.py
  python -m py_compile scripts/run_squat_hailo_auto.py scripts/auto_detect_hailo_bridge.py

Prerequisite: scripts/exercise_detector.py has already been deployed from teammate ZIP.
Set SQUAT_AUTO_TIMEBASE=file for videos (t=frame_id/fps), wall for webcam.
Set SQUAT_AUTO_OUTPUT to a NEW file path. Auto label/session events are separate
from existing REP JSONL to avoid breaking consumers.

No server POST, Good/Bad classifier or session-gated rep counting is added by this patch.
Those are phase two after verifying auto labels and original 20 repetitions.
