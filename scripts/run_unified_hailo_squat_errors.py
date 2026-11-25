#!/usr/bin/env python3
"""Fixed-FPS Hailo smoke test with Bad-only squat error feedback; not a camera runner.
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


def make_error_pipeline(base_class, error_runtime, view):
    """Extend only event publication; preserve counter, buffer and LR180."""
    if view not in ('frontal', 'oblique_profile', 'unknown'):
        raise ValueError('SQUAT_ERROR_VIEW must be frontal, oblique_profile or unknown')

    class ErrorFeedbackPipeline(base_class):
        def _make_event(self, completed_rep):
            event = super()._make_event(completed_rep)
            if event['posture'] != 'bad':
                event.update(error_scores={}, error_status='not_applicable', error_view=None)
                return event
            # Use the same buffered sequence that LR180 sees (with available context).
            # This is not an annotated GT interval; integration is a separate test.
            result = error_runtime.classify(completed_rep.keypoints, 'bad', view)
            event['issues'] = list(dict.fromkeys(event.get('issues', []) + result['issues']))
            event.update(error_scores=result['error_scores'],
                         error_status=result['error_status'], error_view=view)
            return event

    return ErrorFeedbackPipeline


def summarize_error_feedback(path):
    from collections import Counter
    with path.open() as f:
        reps = [json.loads(line) for line in f if line.strip()]
    squat = [e for e in reps if e.get('exercise') == 'squat']
    status = Counter(e['error_status'] for e in squat)
    bad = [e for e in squat if e['posture'] == 'bad']
    invalid = [e for e in squat if (
        (e['posture'] != 'bad' and (e['error_status'] != 'not_applicable' or e['error_scores'] or e['issues']))
        or (e['posture'] == 'bad' and e['error_status'] not in ('identified','undetermined','unassessable','view_unknown'))
        or (e['posture'] == 'bad' and set(e['issues']) != {k for k,v in e['error_scores'].items() if v >= .9})
    )]
    return dict(squat_good=sum(e['posture']=='good' for e in squat), squat_bad=len(bad),
                squat_unknown=sum(e['posture']=='unknown' for e in squat),
                statuses=dict(status), issues=dict(Counter(i for e in bad for i in e['issues'])),
                bad_with_scores=sum(bool(e['error_scores']) for e in bad),
                bad_with_issues=sum(bool(e['issues']) for e in bad),
                contract_pass=not invalid)


def self_test():
    from types import SimpleNamespace
    from tempfile import TemporaryDirectory
    class Base:
        def _make_event(self, rep):
            return dict(exercise='squat', count=7, posture=rep.posture, issues=[], score=None)
    class FakeRuntime:
        def __init__(self): self.calls=[]
        def classify(self, seq, posture, view):
            self.calls.append((seq,posture,view))
            if view=='unknown': return dict(issues=[],error_scores={},error_status='view_unknown')
            return dict(issues=['ERR_KNEES_INSIDE'],error_scores={'ERR_KNEES_INSIDE':.91},error_status='identified')
    rt=FakeRuntime(); cls=make_error_pipeline(Base,rt,'frontal'); pipe=cls()
    seq=np.zeros((30,17,3)); events=[]
    for p in ('good','unknown','bad'):
        e=pipe._make_event(SimpleNamespace(posture=p,keypoints=seq)); events.append(e)
        assert e['posture']==p and e['count']==7
        if p!='bad': assert e['error_status']=='not_applicable' and not e['issues'] and not e['error_scores']
    assert len(rt.calls)==1 and rt.calls[0][0] is seq and events[-1]['issues']==['ERR_KNEES_INSIDE']
    e=make_error_pipeline(Base,rt,'unknown')()._make_event(SimpleNamespace(posture='bad',keypoints=seq)); events.append(e)
    assert e['error_status']=='view_unknown' and not e['issues']
    with TemporaryDirectory() as d:
        path=Path(d)/'reps.jsonl'; path.write_text(''.join(json.dumps(e)+'\n' for e in events))
        stats=summarize_error_feedback(path)
        assert stats['contract_pass'] and stats['squat_bad']==2 and stats['bad_with_issues']==1
    print('PASS: Bad-only event wrapper, unknown view, output contract (mock runtime)')


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
    from squat_error_runtime import SquatErrorRuntime, sha
    view = os.environ.get('SQUAT_ERROR_VIEW','unknown').strip()
    bundle = ROOT/'results/squat_error_deployment/squat_error_numpy.json'
    manifest = json.loads(bundle.with_name('manifest.json').read_text())
    if sha(bundle) != manifest['bundle_sha256'] or sha(ROOT/'scripts/squat_error_runtime.py') != manifest['runtime_sha256']:
        raise RuntimeError('Squat error bundle/runtime hash mismatch')
    error_runtime = SquatErrorRuntime(bundle)
    ErrorPipeline = make_error_pipeline(SquatStreamPipeline, error_runtime, view)
    exercise, squat = models()
    def factory():
        return ErrorPipeline(artifact=squat, fps=fps, bottom_depth=.2,
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
    target_text = os.getenv('UNIFIED_TARGET_TRACK_ID', '').strip()
    target_id = int(target_text) if target_text else None
    if target_id is not None and target_id < 0:
        raise ValueError('Expected nonnegative track ID')
    raw_counts = {'zero': 0, 'single': 0, 'multi': 0}
    target_matches = 0
    selection = os.getenv('SQUAT_PERSON_SELECTION', 'strict')
    if target_id is not None and selection != 'strict':
        raise ValueError('Manual track mode requires SQUAT_PERSON_SELECTION=strict')

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
            if target_id is not None:
                detections = select_track(detections, target_id, hailo.HAILO_UNIQUE_ID)
                target_matches += len(detections)
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
    feedback = summarize_error_feedback(out/'reps.jsonl')
    summary['squat_error_feedback'] = dict(feedback, view=view, threshold=.9,
        bundle_sha256=manifest['bundle_sha256'], sequence='same buffered LR180 interval')
    passed = passed and feedback['contract_pass']
    summary['fixed_fps_smoke_pass']=bool(passed)
    temp = out/'summary.json.tmp'
    temp.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    temp.replace(out/'summary.json')
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    if not passed:
        raise RuntimeError('File integration preconditions failed; inspect summary.json')

if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--self-test']:
        self_test()
    else:
        main()
