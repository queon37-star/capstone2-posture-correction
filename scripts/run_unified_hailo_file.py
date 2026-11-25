#!/usr/bin/env python3
"""Fixed-FPS stored-video Hailo integration smoke test; not a live-camera runner.
Uses the verified generated runtime, existing Hailo adapter and NumPy models.
Hailo's existing CLI parses --input, --hef-path, --disable-sync, etc.
"""
import json
import math
import os
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class UnifiedBridgeAdapter:
    """Adapt push_selected to HailoSquatBridge's push contract."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.push_published = 0
        self.squat_published = 0
        self.unknown_squat = 0
        self.alignment_warnings = 0
        self.push_unassessable = 0

    def _events(self, squat):
        self.squat_published += len(squat)
        self.unknown_squat += sum(e.get('posture') == 'unknown' for e in squat)
        new = self.pipeline.push_events[self.push_published:]
        self.push_published += len(new)
        self.alignment_warnings += sum(e['alignment_flag'] == 1 for e in new)
        self.push_unassessable += sum(e['alignment_flag'] == -1 for e in new)
        # Do not equate absence of an alignment warning with good posture.
        push = [dict(e, type='rep', posture='unknown',
                     issues=['alignment'] if e['alignment_flag'] == 1 else [],
                     score=None) for e in new]
        return clean(list(squat) + push)

    def push(self, kp, *, frame_id):
        k = np.asarray(kp, dtype=np.float32)
        selected = bool(np.isfinite(k[:, :2]).any())
        return self._events(self.pipeline.push_selected(k, selected, frame_id=frame_id))

    def flush(self):
        return self._events(self.pipeline.flush())


def main():
    import sys
    from prepare_board_runtime import models, verify_sources
    from squat_stream_pipeline import SquatStreamPipeline
    from live_components.unified import UnifiedExercisePipeline
    from live_components.detector import ExitHysteresisDetector
    from live_components.squat_classifier import classify
    from hailo_squat_bridge import HailoSquatBridge
    from run_squat_hailo_auto import SquatPoseApp, app_callback_class, get_caps_from_pad, Gst, hailo

    verify_sources()
    fps = float(os.getenv('SQUAT_FPS', '30'))
    if fps not in (15., 30.):
        raise ValueError('Use verified constant 15 or 30 FPS input only')
    out = Path(os.environ['UNIFIED_OUTPUT_DIR']).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['SQUAT_AUTO_TIMEBASE'] = 'file'
    os.environ['SQUAT_SINGLE_PASS'] = '1'
    os.environ['SQUAT_HEADLESS'] = '1'
    examples = Path(os.environ['HAILO_EXAMPLES_ROOT']).expanduser()
    if not os.getenv('HAILO_ENV_FILE') and (examples/'.env').is_file():
        os.environ['HAILO_ENV_FILE'] = str(examples/'.env')
    exercise, squat = models()
    def factory():
        return SquatStreamPipeline(artifact=squat, fps=fps, bottom_depth=.2,
                                   bottom_knee=120, pre_frames=16, post_frames=22,
                                   classifier=classify)
    pipe = UnifiedExercisePipeline(factory(), fps, events_path=out/'sessions.jsonl',
        kp_dump_path='', make_base=factory, gating=1,
        detector=ExitHysteresisDetector(exercise,fps,2.5), replay_sec=6.,
        lying_frames=5, gap_frames=max(10,int(fps*.5)))
    pipe.lost_sec = 3.
    adapter = UnifiedBridgeAdapter(pipe)
    durations, pts_values = [], []
    errors = []
    first_wall = last_wall = None
    expected = int(os.getenv('UNIFIED_EXPECTED_FRAMES','0'))

    class Data(app_callback_class):
        pass

    def callback(pad, info, data):
        nonlocal first_wall, last_wall
        if errors:
            return Gst.PadProbeReturn.OK
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        now = time.perf_counter()
        if first_wall is None:
            first_wall = now
        try:
            data.increment()
            frame_id = data.get_count()-1
            pts_values.append(None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts))
            _, width, height = get_caps_from_pad(pad)
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            tic = time.perf_counter()
            bridge.process(frame_id=frame_id, detections=detections, width=width,
                height=height, landmarks_type=hailo.HAILO_LANDMARKS,
                identity_type=hailo.HAILO_UNIQUE_ID)
            durations.append((time.perf_counter()-tic)*1000.)
            last_wall = time.perf_counter()
            if (frame_id+1)%150==0:
                print('UNIFIED', frame_id+1, 'active',pipe.active_exercise,
                      'squat',adapter.squat_published,'pushup',adapter.push_published,flush=True)
        except Exception as e:
            # Pad probes may otherwise swallow exceptions and continue silently.
            errors.append(repr(e))
            print('CALLBACK ERROR',repr(e),file=sys.stderr,flush=True)
        return Gst.PadProbeReturn.OK

    wall_start = time.perf_counter()
    with (out/'reps.jsonl').open('x',encoding='utf-8') as writer:
        bridge = HailoSquatBridge(adapter,writer,person_selection=os.getenv('SQUAT_PERSON_SELECTION','strict'))
        data = Data()
        try:
            app = SquatPoseApp(callback,data)
            if app.source_type != 'file':
                raise ValueError('Stored-video source only')
            app.run()
        finally:
            bridge.finish()
    n = bridge.processed_frames
    d = np.asarray(durations,float)
    pts_ok = len(pts_values)==n and n>1 and all(t is not None for t in pts_values)
    delta = np.diff(np.array(pts_values,dtype=np.int64))/1e9 if pts_ok else np.array([])
    # Check that callback order really represents the fixed source clock.
    bad_pts = int(np.count_nonzero(~np.isclose(delta,1/fps,atol=.001,rtol=.02))) if pts_ok else None
    summary = dict(frames=n,expected_frames=expected or None,source_fps=fps,
        valid_pose_frames=bridge.valid_person_frames,missing_caps_frames=bridge.missing_caps_frames,
        squat_count=adapter.squat_published,pushup_count=adapter.push_published,
        squat_unknown=adapter.unknown_squat,pushup_alignment_warnings=adapter.alignment_warnings,
        pushup_unassessable=adapter.push_unassessable,
        callback_fps=(n-1)/(last_wall-first_wall) if n>1 and last_wall>first_wall else None,
        wall_sec=time.perf_counter()-wall_start,
        bridge_ms={k:float(v) for k,v in zip(('mean','p50','p95','p99','max'),[d.mean(),*np.percentile(d,[50,95,99]),d.max()])} if len(d) else {},
        pts_available=pts_ok,pts_bad_intervals=bad_pts,callback_errors=errors,
        note='Bridge timing includes pose selection, runtime and event output; not camera-to-feedback latency. Counts are smoke-test outputs, not independent accuracy.')
    passed = n>0 and bridge.valid_person_frames>0 and not errors and pts_ok and bad_pts==0 and (not expected or expected==n)
    summary['fixed_fps_smoke_pass']=bool(passed)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    if not passed:
        raise RuntimeError('File integration preconditions failed; inspect summary.json')

if __name__=='__main__':
    main()
