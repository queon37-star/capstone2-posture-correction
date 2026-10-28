"""Hailo GStreamer YOLOv8-Pose -> squat repetition JSONL proof-of-integration.

Run on the R2130-12 / supported Hailo runtime, not on a normal GPU PC.
Derived from hailo-ai/hailo-rpi5-examples/basic_pipelines/pose_estimation.py:
same GStreamerPoseEstimationApp, pad-probe callback, HAILO_DETECTION, caps.
Supports the installed R2130-12 hailo_apps_infra API and newer hailo_apps imports.

Usage from repository root (the trusted model file must already be copied):
    export HAILO_EXAMPLES_ROOT="$HOME/hailo-rpi5-examples"  # adjust
    export SQUAT_COUNT_ONLY=1  # No joblib or sklearn needed for counter smoke test
    # For classification later: unset SQUAT_COUNT_ONLY and set SQUAT_MODEL_PATH.
    export SQUAT_OUTPUT="results/hailo/live_reps.jsonl"
    export SQUAT_FPS="30"
    export SQUAT_SINGLE_PASS=1  # stop automatically at video EOS
    export SQUAT_BATCH_SIZE=4   # optional; default stays 2
    export SQUAT_PACE_FILE=1     # optional: pace timestamped file at playback rate
    python scripts/run_squat_hailo_pose.py

This script writes events to JSONL only. It does NOT send HTTP POST to FastAPI.
"""

import os
import sys
import time
from pathlib import Path

# Allow use of Hailo's checked-out examples repo without copying this script.
examples_root = os.getenv('HAILO_EXAMPLES_ROOT')
if examples_root:
    sys.path.insert(0, str(Path(examples_root).expanduser().resolve()))

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import hailo

# R2130-12 HailoRT 4.20.0 examples use hailo_apps_infra.
try:
    from hailo_apps_infra.hailo_rpi_common import get_caps_from_pad, app_callback_class
    from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp
except ImportError:
    from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad
    from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import (
        GStreamerPoseEstimationApp,
    )

from hailo_squat_bridge import HailoSquatBridge
from squat_stream_pipeline import SquatStreamPipeline


class SquatPoseApp(GStreamerPoseEstimationApp):
    """Optional one-pass playback and no-overlay sink for hardware validation."""

    def get_pipeline_string(self):
        # Hailo's base __init__ sets batch_size=2 and builds the pipeline
        # immediately, so override here *before* constructing its hailonet.
        requested_batch = os.getenv('SQUAT_BATCH_SIZE')
        if requested_batch is not None:
            try:
                batch_size = int(requested_batch)
            except ValueError as exc:
                raise ValueError('SQUAT_BATCH_SIZE must be 1, 2, or 4') from exc
            if batch_size not in (1, 2, 4):
                raise ValueError('SQUAT_BATCH_SIZE must be 1, 2, or 4')
            self.batch_size = batch_size
        print(f'Hailo Pose batch_size={self.batch_size}', flush=True)
        pipeline = super().get_pipeline_string()

        if os.getenv('SQUAT_PACE_FILE', '0') == '1':
            if self.source_type != 'file':
                raise ValueError('SQUAT_PACE_FILE=1 is for timestamped video files only')
            # Use the actual installed source builder and verify its exact
            # prefix before inserting a clock-synchronizing identity element.
            source_fn = GStreamerPoseEstimationApp.get_pipeline_string.__globals__['SOURCE_PIPELINE']
            source_pipeline = source_fn(
                video_source=self.video_source,
                video_width=self.video_width,
                video_height=self.video_height,
            )
            prefix = source_pipeline + ' !'
            if not pipeline.startswith(prefix):
                raise RuntimeError('Unexpected Hailo source pipeline; cannot safely insert file pacer')
            pipeline = (
                prefix + ' identity name=squat_file_pacer sync=true signal-handoffs=false ! '
                + pipeline[len(prefix):]
            )
            print('SQUAT_PACE_FILE=1: pacing frames by file timestamps.', flush=True)

        if os.getenv('SQUAT_HEADLESS', '0') != '1':
            return pipeline

        # The actual installed pose class owns the DISPLAY_PIPELINE helper.
        # Retrieve that exact helper so the tail matches the installed API,
        # rather than assuming a hard-coded pipeline string.
        display_fn = GStreamerPoseEstimationApp.get_pipeline_string.__globals__['DISPLAY_PIPELINE']
        original_display = display_fn(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
        )
        if not pipeline.endswith(original_display):
            raise RuntimeError('Unexpected Hailo display pipeline; refusing unsafe replacement')

        # Keep inference, tracker and user callback; remove only display
        # overlay, extra colorspace conversion and GUI rendering.
        headless = pipeline[:-len(original_display)] + (
            'fakesink name=squat_headless_sink sync=false async=false'
        )
        print('SQUAT_HEADLESS=1: bypass overlay, display convert and fpsdisplaysink.')
        print('Effective pipeline:', headless, flush=True)
        return headless

    def on_eos(self):
        # The installed GStreamerApp rewinds file sources by default.
        # For reproducible dataset evaluation, finish exactly one pass.
        if os.getenv('SQUAT_SINGLE_PASS', '0') == '1' and self.source_type == 'file':
            print('EOS: file finished; stopping without rewind.', flush=True)
            self.shutdown()
        else:
            super().on_eos()


class SquatCallbackData(app_callback_class):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.started_at = time.perf_counter()
        self.first_callback_at = None
        self.first_frame_pts = None
        self.pace_drift_ms = []


def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    if os.getenv('SQUAT_PACE_FILE', '0') == '1':
        # Measure relative pace drift, not absolute camera-to-feedback latency.
        pts = buffer.pts
        if pts != Gst.CLOCK_TIME_NONE:
            now = time.perf_counter()
            if user_data.first_frame_pts is None:
                user_data.first_frame_pts = pts
                user_data.first_callback_at = now
            file_elapsed_ms = (pts - user_data.first_frame_pts) / 1_000_000
            callback_elapsed_ms = (now - user_data.first_callback_at) * 1000
            user_data.pace_drift_ms.append(callback_elapsed_ms - file_elapsed_ms)

    # The frame count starts at 1 after increment(); use zero-based frame IDs.
    user_data.increment()
    frame_id = user_data.get_count() - 1

    _format, width, height = get_caps_from_pad(pad)
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    user_data.bridge.process(
        frame_id=frame_id, detections=detections,
        width=width, height=height,
        landmarks_type=hailo.HAILO_LANDMARKS,
    )

    if (frame_id + 1) % 150 == 0:
        bridge = user_data.bridge
        elapsed = max(time.perf_counter() - user_data.started_at, 1e-6)
        print(
            f'frame={frame_id} state={bridge.pipeline.counter.state} '
            f'detected_reps={bridge.pipeline.counter.count} '
            f'published_reps={bridge.pipeline.event_count} '
            f'valid_pose={bridge.valid_person_frames}/{bridge.processed_frames} '
            f'persons_0_1_multi={bridge.zero_person_frames}/'
            f'{bridge.single_person_frames}/{bridge.multi_person_frames} '
            f'single_missing_pose={bridge.single_person_without_pose_frames} '
            f'missing_caps={bridge.missing_caps_frames} '
            f'callback_fps={bridge.processed_frames/elapsed:.1f}',
            flush=True,
        )
    return Gst.PadProbeReturn.OK


def main():
    project_root = Path(__file__).resolve().parents[1]
    count_only = os.getenv('SQUAT_COUNT_ONLY', '0') == '1'
    model_path = Path(os.getenv(
        'SQUAT_MODEL_PATH', 'models/squat_plus_asymmetry180_crop10_all9.joblib'
    )).expanduser()
    if not model_path.is_absolute():
        model_path = project_root / model_path
    output_path = Path(os.getenv('SQUAT_OUTPUT', 'results/hailo/live_reps.jsonl')).expanduser()
    if not output_path.is_absolute():
        output_path = project_root / output_path
    fps = float(os.getenv('SQUAT_FPS', '30'))
    if not 0 < fps <= 240:
        raise ValueError('SQUAT_FPS must be in (0, 240]')
    artifact = None
    if not count_only:
        if not model_path.is_file():
            raise FileNotFoundError(f'Missing classifier artifact: {model_path}')
        # Full classifier mode only: require compatible joblib/sklearn.
        import joblib
        artifact = joblib.load(model_path)
        if artifact.get('feature_variant') != 'PlusAsymmetry180':
            raise ValueError('Expected PlusAsymmetry180 classifier artifact')
    pipeline = SquatStreamPipeline(
        artifact=artifact, fps=fps, bottom_depth=0.20,
        bottom_knee=120.0, pre_frames=16, post_frames=22,
        count_only=count_only,
    )

    # The original Hailo sample sets HAILO_ENV_FILE to its examples root/.env.
    # Retain an existing caller-provided value; infer it only when available.
    if not os.getenv('HAILO_ENV_FILE') and examples_root:
        env_file = Path(examples_root).expanduser().resolve() / '.env'
        if env_file.is_file():
            os.environ['HAILO_ENV_FILE'] = str(env_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Mode={"COUNT_ONLY" if count_only else "CLASSIFIER"} | assumed_fps={fps:g} | output={output_path}')
    if count_only:
        print('Count-only validation: posture=unknown is EXPECTED; no joblib/sklearn loaded.')
    else:
        print(f'Classifier={model_path}; count and Good/Bad events saved to JSONL.')
    print(
        'Single-person input only; no server POST; '
        f'single_pass={os.getenv("SQUAT_SINGLE_PASS", "0") == "1"} '
        f'headless={os.getenv("SQUAT_HEADLESS", "0") == "1"}.'
    )

    with output_path.open('w', encoding='utf-8') as writer:
        bridge = HailoSquatBridge(pipeline, writer)
        user_data = SquatCallbackData(bridge)
        try:
            app = SquatPoseApp(app_callback, user_data)
            app.run()
        finally:
            bridge.finish()
            elapsed = max(time.perf_counter() - user_data.started_at, 1e-6)
            if os.getenv('SQUAT_PACE_FILE', '0') == '1':
                if user_data.pace_drift_ms:
                    drifts = sorted(user_data.pace_drift_ms)
                    p95 = drifts[int(0.95 * (len(drifts) - 1))]
                    print(
                        f'FILE_PACE: timestamped={len(drifts)}/{bridge.processed_frames} '
                        f'final_relative_drift={user_data.pace_drift_ms[-1]:.1f}ms '
                        f'p95_relative_drift={p95:.1f}ms '
                        f'max_relative_drift={drifts[-1]:.1f}ms '
                        '(relative to first callback; NOT end-to-end latency)',
                        flush=True,
                    )
                else:
                    print('FILE_PACE: no valid buffer timestamps; check file source', flush=True)
            print(
                f'DONE: {bridge.processed_frames} frames, '
                f'{pipeline.event_count} complete reps, '
                f'valid_pose={bridge.valid_person_frames}/{bridge.processed_frames or 1}, '
                f'persons_0_1_multi={bridge.zero_person_frames}/'
                f'{bridge.single_person_frames}/{bridge.multi_person_frames}, '
                f'single_missing_pose={bridge.single_person_without_pose_frames}, '
                f'missing_caps={bridge.missing_caps_frames}, '
                f'wall={elapsed:.1f}s, callback_fps={bridge.processed_frames/elapsed:.1f}, '
                f'output={output_path}',
                flush=True,
            )


if __name__ == '__main__':
    main()
