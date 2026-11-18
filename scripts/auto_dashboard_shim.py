"""Non-invasive Hailo auto-detection + existing browser dashboard integration.

This module is loaded *before* the project's run_squat_pose_client_render.py.
It wraps one existing SquatStreamPipeline instance with AutoDetectPipeline;
no second pose inference, no second person selection, no threshold changes.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

_STATE = {"pipeline": None, "started_at": time.perf_counter(), "hook_used": False,
          "classifier_preload": "not_requested"}
_SERVER = None


def _path_from_env(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default)).expanduser()
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parents[1] / p


def _read_latest_rep():
    path = _path_from_env("SQUAT_OUTPUT", "results/hailo/live_reps.jsonl")
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            length = f.tell()
            f.seek(max(0, length - 131072))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    return {key: event.get(key) for key in (
                        "type", "frame_id", "rep_index", "rep_no", "count", "posture",
                        "score", "confidence", "reason", "duration_s") if key in event}
            except (ValueError, TypeError):
                continue
    except (OSError, PermissionError):
        pass
    return None


def _status():
    p = _STATE["pipeline"]
    result = {
        "running": p is not None,
        "hook_used": _STATE["hook_used"],
        "exercise": None, "raw_exercise": None, "active_session": None,
        "frame_id": None, "counter_state": None, "squat_count": 0,
        "published_reps": 0, "callback_fps": None, "valid_pose": None,
        "last_rep": _read_latest_rep(),
        "count_only": os.getenv("SQUAT_COUNT_ONLY", "0") == "1",
        "timebase": os.getenv("SQUAT_AUTO_TIMEBASE", "wall"),
        "classifier_preload": _STATE["classifier_preload"],
        "kp_dump_frames": None, "gating": None,
    }
    if p is None:
        return result
    detector = p.detector
    counter = p.counter          # gating: cumulative view; otherwise the wrapped counter
    result.update({
        "exercise": p.last_confirmed,
        "raw_exercise": detector.last_raw,
        "active_session": p.active_exercise,
        "frame_id": getattr(p, "dashboard_last_frame", None),
        "counter_state": getattr(counter, "state", None),
        "squat_count": getattr(counter, "count", 0),
        "published_reps": p.event_count,
        "callback_fps": round(getattr(p, "dashboard_frames", 0) /
                               max(time.perf_counter() - p.dashboard_start, .001), 1),
        "valid_pose": getattr(p, "dashboard_valid_pose", None),
        "kp_dump_frames": p.kp_dump.frames if p.kp_dump is not None else None,
        "gating": p.gating,
    })
    if p.gating:
        result.update({
            "session_id": p.current_session_id,
            "session_count": p.session_count if p.active_exercise == "squat" else None,
            "total_count": p.total_count,
            "session_last_rep": p.session_last_rep,
            "last_replay": p.last_replay,
            "max_replay_ms": round(p.max_replay_ms, 2),
        })
    return result


_HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hailo 통합 운동 모니터</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,"Malgun Gothic",sans-serif}
*{box-sizing:border-box}body{background:#0c111c;color:#f1f5f9;margin:0}
header{padding:17px 20px;border-bottom:1px solid #293447;display:flex;align-items:center;justify-content:space-between;gap:12px}
header h1{font-size:18px;margin:0}header p{font-size:12px;color:#9daec6;margin:4px 0 0}
#indicator{font-size:12px;color:#9daec6}
main{display:grid;grid-template-columns:minmax(0,2.4fr) minmax(265px,.95fr);gap:14px;padding:14px;min-height:calc(100vh - 74px)}
.video{background:#080e19;border:1px solid #293447;border-radius:12px;overflow:hidden;min-height:600px;display:flex;flex-direction:column}
.video iframe{border:0;width:100%;height:calc(100vh - 158px);min-height:540px;background:#080e19}
.video footer{font-size:12px;color:#a3b4cd;padding:9px 12px;border-top:1px solid #293447}
a{color:#93c5fd}
.panel{display:flex;flex-direction:column;gap:12px}
.card{border:1px solid #293447;border-radius:12px;background:#131d2e;padding:15px}
.card h2{font-weight:600;font-size:12px;letter-spacing:.05em;color:#a6bad6;margin:0 0 10px}
.big{font-weight:750;font-size:35px;line-height:1.2} .value{font-size:21px;font-weight:700}
.sub{font-size:12px;color:#94a9c4;line-height:1.6;margin-top:8px}
.row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;gap:10px;border-bottom:1px solid #293447;font-size:13px}
.row:last-child{border-bottom:0}.muted{color:#a3b4cd}
.good{color:#6ee7b7}.bad{color:#fda4af}.warn{color:#fbbf24}
@media(max-width:1000px){main{grid-template-columns:1fr}.video{min-height:0}.video iframe{height:68vh;min-height:450px}}
</style></head><body>
<header><div><h1>운동 자세 교정 · 통합 실시간 모니터</h1><p>Hailo-8 Pose / ExerciseDetector / Squat Good-Bad</p></div><span id="indicator">연결 중…</span></header>
<main><section class="video"><iframe id="viewer" title="기존 C922 실시간 영상"></iframe>
<footer>기존 카메라·스켈레톤 화면 · <a id="direct" target="_blank" rel="noopener">영상만 별도 열기</a> · 기존 화면이 나타나지 않으면 영상 링크를 확인</footer></section>
<aside class="panel">
 <section class="card"><h2>자동 운동 종류 · 확정 판정</h2><div class="big" id="exercise">판별 대기</div><div class="sub">즉시 판정: <span id="raw">—</span> · 활성 세션: <span id="session">—</span></div></section>
 <section class="card"><h2>스쿼트 카운팅</h2><div class="big" id="reps">0</div><div class="row"><span class="muted">완료 이벤트(누적)</span><b id="published">0</b></div><div class="row"><span class="muted">현재 세션 반복</span><b id="sessreps">—</b></div><div class="row"><span class="muted">카운터 상태</span><b id="state">—</b></div></section>
 <section class="card"><h2>최근 완료 스쿼트 · Good/Bad</h2><div class="value" id="posture">대기 중</div><div class="sub" id="score">완료된 반복이 기록되면 결과 표시</div></section>
 <section class="card"><h2>처리 상태</h2><div class="row"><span class="muted">Callback (참고)</span><b id="fps">—</b></div><div class="row"><span class="muted">Frame</span><b id="frame">—</b></div><div class="row"><span class="muted">Pose 사용 가능</span><b id="pose">—</b></div><div class="row"><span class="muted">자동 판별 연결</span><b id="hook">—</b></div><div class="sub">Callback FPS는 노트북 영상 FPS나 전체 지연 시간이 아님</div></section>
</aside></main>
<script>
const p=new URLSearchParams(location.search);const oldPort=p.get('video_port')||'18765';
const oldUrl=location.protocol+'//'+location.hostname+':'+oldPort+'/';
document.getElementById('viewer').src=oldUrl;document.getElementById('direct').href=oldUrl;
const $=x=>document.getElementById(x);
function label(v){return v==='squat'?'SQUAT':v==='pushup'?'PUSH-UP':v==='idle'?'IDLE':'판별 대기';}
async function update(){try{
 const r=await fetch('/status',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const s=await r.json();
 $('indicator').textContent=s.running?'자동 판별 연결됨':'파이프라인 시작 대기';
 $('exercise').textContent=label(s.exercise);$('raw').textContent=label(s.raw_exercise);
 $('session').textContent=s.active_session?label(s.active_session):'없음';
 $('reps').textContent=s.squat_count??0;$('published').textContent=s.published_reps??0;
 $('sessreps').textContent=s.gating?(s.session_count==null?'세션 없음':s.session_count+' (세션 #'+s.session_id+')'):'게이팅 꺼짐';
 $('state').textContent=s.counter_state??'—'; $('fps').textContent=s.callback_fps==null?'—':s.callback_fps+' FPS';
 $('frame').textContent=s.frame_id??'—';$('pose').textContent=s.valid_pose===null?'—':(s.valid_pose?'YES':'NO');
 $('hook').textContent=s.hook_used?'YES':'대기';
 // gating: only this run's in-memory session result (no fallback to an older JSONL line)
 let rep=s.gating?(s.session_last_rep||{}):(s.last_rep||{});let posture=rep.posture;
 $('posture').textContent=s.count_only?'분류 비활성화':posture==='good'?'GOOD':posture==='bad'?'BAD':posture==='unknown'?'UNKNOWN':posture?String(posture):'대기 중';
 $('posture').className='value '+(posture==='good'?'good':posture==='bad'?'bad':'');
 let parts=[];if(rep.score!==undefined&&rep.score!==null)parts.push('Score: '+rep.score);
 if(rep.frame_id!==undefined)parts.push('Frame: '+rep.frame_id);
 if(s.gating&&rep.session_id!=null)parts.push((s.active_session==='squat'&&rep.session_id===s.session_id?'현재 세션 #':'이전 세션 #')+rep.session_id);
 if(rep.pre_context_complete===false)parts.push('pre-context '+rep.pre_context_frames+'f ('+rep.pre_context_cut+')');
 $('score').textContent=parts.length?parts.join(' · '):'완료된 반복이 기록되면 결과 표시';
 }catch(e){$('indicator').textContent='상태 연결 실패: '+e.message;}}
update();setInterval(update,600);
</script></body></html>'''


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/status":
            body = json.dumps(_status(), ensure_ascii=False, default=str).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif route == "/" or route == "/index.html":
            body = _HTML.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def start_status_server():
    global _SERVER
    port = int(os.getenv("SQUAT_AUTO_DASHBOARD_PORT", "8766"))
    # loopback binding: reach via SSH forwarding; do not expose on local LAN.
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.daemon_threads = True
    _SERVER = server
    threading.Thread(target=server.serve_forever, name="auto_dashboard", daemon=True).start()
    print(f"AUTO_DASHBOARD listening on 127.0.0.1:{port}", flush=True)


def preload_classifier():
    """Import the Good/Bad feature/classifier modules before inference starts.

    SquatStreamPipeline imports squat_classification lazily inside the first
    completed repetition, i.e. inside the Hailo callback; that pulls in pandas,
    matplotlib and sklearn (about 1.2 s on a desktop CPU). Importing here moves
    that one-time cost before the camera starts. Count-only mode skips it so no
    classifier dependency is added. Outputs are unchanged: an import failure is
    reported here and still recorded per repetition exactly as before.
    """
    if os.getenv("SQUAT_COUNT_ONLY", "0") == "1":
        _STATE["classifier_preload"] = "skipped_count_only"
        return
    start = time.perf_counter()
    try:
        import squat_classification  # noqa: F401  (cached in sys.modules)
    except Exception as exc:  # keep the original per-rep error behaviour
        _STATE["classifier_preload"] = f"error: {type(exc).__name__}: {exc}"
        print(f"WARNING: classifier preload failed ({type(exc).__name__}: {exc}); "
              "Good/Bad events will carry classification_error.", flush=True)
        return
    ms = 1000 * (time.perf_counter() - start)
    _STATE["classifier_preload"] = f"ok ({ms:.0f} ms)"
    print(f"AUTO_DASHBOARD classifier preloaded in {ms:.0f} ms", flush=True)


def install_pipeline_hook():
    preload_classifier()
    import squat_stream_pipeline
    from auto_detect_hailo_bridge import AutoDetectPipeline
    original_pipeline_class = squat_stream_pipeline.SquatStreamPipeline

    class DashboardAutoPipeline(AutoDetectPipeline):
        def __init__(self, base, fps, **kwargs):
            super().__init__(base, fps, **kwargs)
            self.dashboard_start = time.perf_counter()
            self.dashboard_frames = 0
            self.dashboard_last_frame = None
            self.dashboard_valid_pose = None
            _STATE["pipeline"] = self
            _STATE["hook_used"] = True

        def push(self, kp, *, frame_id, t=None):
            result = super().push(kp, frame_id=frame_id, t=t)
            self.dashboard_frames += 1
            self.dashboard_last_frame = frame_id
            # A value measure() can be computed from keypoints; not a new inference.
            self.dashboard_valid_pose = bool(self.detector.buf and self.detector.buf[-1][1] is not None)
            return result

    def _pipeline_factory(*args, **kwargs):
        original = original_pipeline_class(*args, **kwargs)
        fps = kwargs.get("fps")
        if fps is None:
            fps = os.getenv("SQUAT_FPS", "30")
        # Same constructor arguments (artifact, thresholds, count_only, ...) for
        # every new squat session when SQUAT_AUTO_GATING=1.
        return DashboardAutoPipeline(
            original, float(fps),
            make_base=lambda: original_pipeline_class(*args, **kwargs))

    squat_stream_pipeline.SquatStreamPipeline = _pipeline_factory
    print("AUTO_DASHBOARD pipeline hook installed (existing Hailo inference reused)", flush=True)
