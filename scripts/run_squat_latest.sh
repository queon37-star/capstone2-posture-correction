#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec python scripts/run_squat_video_stream_tunable.py \
    --model models/squat_plus_asymmetry180_crop10_all9.joblib \
    --bottom-depth 0.20 \
    --pre-frames 16 \
    --post-frames 22 \
    "$@"
