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


def select_track(detections, track_id, identity_type):
    """File-specific manual target: fail closed on absence or ambiguity."""
    matches = []
    for det in detections:
        if det.get_label() != 'person':
            continue
        ids = det.get_objects_typed(identity_type)
        if len(ids) == 1 and int(ids[0].get_id()) == track_id:
            matches.append(det)
    return matches if len(matches) == 1 else []


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


def dominant_person(detections, identity_type):
    """Diagnostic common policy: choose unique largest visible box only with margin.
    No clip labels, boundary information or manually supplied IDs. Not identity tracking.
    """
    ranked=[]
    for d in detections:
        if d.get_label()!='person': continue
        b=d.get_bbox(); x,y,w,h=map(float,(b.xmin(),b.ymin(),b.width(),b.height()))
        if not np.isfinite([x,y,w,h]).all() or w<=0 or h<=0: continue
        area=max(0,min(1,x+w)-max(0,x))*max(0,min(1,y+h)-max(0,y))
        if area>0: ranked.append((area,d))
    ranked.sort(key=lambda q:q[0],reverse=True)
    if not ranked or (len(ranked)>1 and ranked[0][0]<1.5*ranked[1][0]): return [],None
    d=ranked[0][1]; ids=d.get_objects_typed(identity_type)
    return [d],int(ids[0].get_id()) if len(ids)==1 else None


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
    run_errors = []
    eos_seen = False
    exit_code = None
    first_wall = last_wall = None
    expected = int(os.getenv('UNIFIED_EXPECTED_FRAMES','0'))
    frame_trace = []
    target_text = os.getenv('UNIFIED_TARGET_TRACK_ID', '').strip()
    target_id = int(target_text) if target_text else None
    if target_id is not None and target_id < 0:
        raise ValueError('Expected nonnegative track ID')
    raw_counts = {'zero': 0, 'single': 0, 'multi': 0}
    target_matches = 0
    selection = os.getenv('SQUAT_PERSON_SELECTION', 'strict')
    if target_id is not None or selection != 'strict':
        raise ValueError('Mixed runner requires no target ID and strict bridge; common dominant selector is applied before bridge')

    class Data(app_callback_class):
        pass

    class MeasuredPoseApp(SquatPoseApp):
        def on_eos(self):
            nonlocal eos_seen
            eos_seen = True
            return super().on_eos()

    def callback(pad, info, data):
        nonlocal first_wall, last_wall, target_matches
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
            people = sum(det.get_label() == 'person' for det in detections)
            raw_counts['zero' if people == 0 else 'single' if people == 1 else 'multi'] += 1
            detections, selected_id = dominant_person(detections, hailo.HAILO_UNIQUE_ID)
            before_valid = bridge.valid_person_frames
            bridge.process(frame_id=frame_id, detections=detections, width=width,
                height=height, landmarks_type=hailo.HAILO_LANDMARKS,
                identity_type=hailo.HAILO_UNIQUE_ID)
            frame_trace.append(dict(frame_id=frame_id, raw=pipe.detector.last_raw,
                confirmed=pipe.detector.confirmed, active=pipe.active_exercise,
                raw_people=people, selected_track_id=selected_id,
                valid_pose=bridge.valid_person_frames>before_valid))
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
        bridge = HailoSquatBridge(adapter,writer,person_selection=selection)
        data = Data()
        try:
            app = MeasuredPoseApp(callback,data)
            if app.source_type != 'file':
                raise ValueError('Stored-video source only')
            app.run()
        except SystemExit as exc:
            # Installed Hailo shutdown can exit Python rather than return.
            exit_code = exc.code
            if not eos_seen or exc.code not in (None, 0):
                run_errors.append(f'Unexpected SystemExit: {exc.code!r}; eos_seen={eos_seen}')
        except KeyboardInterrupt:
            run_errors.append('KeyboardInterrupt before normal completion')
        except Exception as exc:
            run_errors.append(repr(exc))
        finally:
            try:
                bridge.finish()
            except Exception as exc:
                run_errors.append('bridge.finish: '+repr(exc))
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
        eos_seen=eos_seen, exit_code=clean(exit_code) if isinstance(exit_code,(int,str,type(None))) else repr(exit_code),
        run_errors=run_errors,
        note='Bridge timing includes pose selection, runtime and event output; not camera-to-feedback latency. Counts are smoke-test outputs, not independent accuracy.')
    passed = eos_seen and not run_errors and n>0 and bridge.valid_person_frames>0 and not errors and pts_ok and bad_pts==0 and (not expected or expected==n)
    summary.update(target_track_id=target_id, person_selection=selection,
        raw_person_frames=raw_counts,
        target_match_frames=target_matches if target_id is not None else None,
        target_missing_or_ambiguous_frames=n-target_matches if target_id is not None else None)
    summary['person_selection']='dominant_area_margin_1.5'
    with (out/'frame_trace.jsonl').open('x') as f:
        for row in frame_trace: f.write(json.dumps(clean(row),allow_nan=False)+'\n')
    summary['fixed_fps_smoke_pass']=bool(passed)
    temp = out/'summary.json.tmp'
    temp.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    temp.replace(out/'summary.json')
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    if not passed:
        raise RuntimeError('File integration preconditions failed; inspect summary.json')

if __name__=='__main__':
    main()
