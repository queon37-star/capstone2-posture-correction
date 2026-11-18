"""C922 + Hailo live raw-camera + pose/count metadata; overlay rendered in laptop browser.

Place beside run_squat_hailo_pose.py in ~/capstone2/pose_training/scripts.
Run with the same environment variables and Hailo --input/--arch/--hef-path arguments
as run_squat_hailo_pose.py.  The board only copies an occasional raw frame,
encodes JPEG on a separate worker and transmits it with pose JSON.  All
keypoint lines, labels, boxes, and counts are DRAWN IN THE BROWSER (Canvas).

Video JPEG and pose metadata are carried in a single SSE JSON packet, with the
same frame_id, to avoid incorrect overlay/video synchronization.
"""
import base64
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import hailo
import numpy as np
from hailo_apps_infra.hailo_rpi_common import get_numpy_from_buffer
import run_squat_hailo_pose as original

PORT = int(os.getenv('SQUAT_DASHBOARD_PORT', '8765'))
PREVIEW_FPS = float(os.getenv('SQUAT_PREVIEW_FPS', '10'))
PREVIEW_WIDTH = int(os.getenv('SQUAT_PREVIEW_WIDTH', '800'))
JPEG_QUALITY = int(os.getenv('SQUAT_JPEG_QUALITY', '72'))
if not 0 < PREVIEW_FPS <= 60:
    raise ValueError('SQUAT_PREVIEW_FPS should be in (0,60]')
if not 160 <= PREVIEW_WIDTH <= 1920:
    raise ValueError('SQUAT_PREVIEW_WIDTH should be 160..1920')
if not 20 <= JPEG_QUALITY <= 95:
    raise ValueError('SQUAT_JPEG_QUALITY should be 20..95')

# Latest-only queue: intentionally discard obsolete frames to prevent backlog.
class LatestSlot:
    def __init__(self):
        self.cv = threading.Condition()
        self.seq = 0
        self.value = None

    def publish(self, value):
        with self.cv:
            self.seq += 1
            self.value = value
            self.cv.notify_all()

    def next_after(self, seq, timeout=3):
        with self.cv:
            changed = self.cv.wait_for(lambda: self.seq != seq, timeout=timeout)
            return (self.seq, self.value) if changed else (seq, None)

raw_slot = LatestSlot()
encoded_slot = LatestSlot()
stop_signal = threading.Event()
callback_times = deque(maxlen=120)
original_callback = original.app_callback
last_preview_at = 0.0
warned = False

# 17 COCO keypoints, in the pixel coordinate system of Gst callback image.
def extract_poses(buffer, width, height):
    persons = []
    detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)
    for det in detections:
        if det.get_label() != 'person':
            continue
        landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        bb = det.get_bbox()
        points = landmarks[0].get_points()
        joints = []
        for i in range(min(17, len(points))):
            p = points[i]
            x = (p.x() * bb.width() + bb.xmin()) * width
            y = (p.y() * bb.height() + bb.ymin()) * height
            joints.append([round(x, 2), round(y, 2)])
        ids = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        track = ids[0].get_id() if ids else None
        persons.append({'track_id': track, 'joints': joints})
    return persons


def pose_callback(pad, info, user_data):
    global warned, last_preview_at
    # Count and JSONL output remain entirely in the original code.
    result = original_callback(pad, info, user_data)
    now = time.monotonic()
    callback_times.append(now)

    # Export <= PREVIEW_FPS frames/s; never delay the inference callback for JPEG.
    if now - last_preview_at < 1.0 / PREVIEW_FPS:
        return result
    last_preview_at = now
    try:
        buffer = info.get_buffer()
        if buffer is None:
            return result
        fmt, width, height = original.get_caps_from_pad(pad)
        if fmt is None or not width or not height:
            return result
        frame = get_numpy_from_buffer(buffer, fmt, width, height)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f'Unexpected pixel data: format={fmt}, shape={getattr(frame, "shape", None)}')
        bridge = user_data.bridge
        fps = ((len(callback_times) - 1) / (callback_times[-1] - callback_times[0])
               if len(callback_times) > 1 and callback_times[-1] > callback_times[0]
               else 0.0)
        state = {
            'frame_id': user_data.get_count() - 1,
            'source_width': int(width), 'source_height': int(height),
            'persons': extract_poses(buffer, width, height),
            'count': int(bridge.pipeline.counter.count),
            'published': int(bridge.pipeline.event_count),
            'state': str(bridge.pipeline.counter.state),
            'callback_fps': round(fps, 2),
            'active_id': (bridge.selector.active_track_id if bridge.selector else None),
        }
        # The Gst buffer may be recycled when this function returns.
        raw_slot.publish((np.ascontiguousarray(frame.copy()), str(fmt).upper(), state))
    except Exception as exc:
        if not warned:
            print('DASHBOARD EXPORT ERROR:', type(exc).__name__, str(exc), flush=True)
            warned = True
    return result


def encoder_worker():
    seq = 0
    while not stop_signal.is_set():
        seq, item = raw_slot.next_after(seq, timeout=1)
        if item is None:
            continue
        try:
            frame, fmt, state = item
            if fmt == 'RGB':
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif fmt == 'BGR':
                img = frame
            else:
                raise ValueError(f'Unsupported Gst format for preview: {fmt}')
            h, w = img.shape[:2]
            if w > PREVIEW_WIDTH:
                new_h = max(1, round(h * PREVIEW_WIDTH / w))
                img = cv2.resize(img, (PREVIEW_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise RuntimeError('JPEG encoder returned false')
            packet = dict(state)
            packet['jpeg_base64'] = base64.b64encode(jpeg.tobytes()).decode('ascii')
            packet['video_width'] = img.shape[1]
            packet['video_height'] = img.shape[0]
            packet['encoded_at_ms'] = round(time.time() * 1000)
            encoded_slot.publish(json.dumps(packet, separators=(',', ':')))
        except Exception as exc:
            print('DASHBOARD ENCODER ERROR:', type(exc).__name__, str(exc), flush=True)


PAGE = r'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C922 / Hailo-8 / Squat Live</title>
<style>
body {margin:0;background:#11151b;color:#f1f5f9;font-family:system-ui,-apple-system,sans-serif}
main {max-width:1100px;margin:auto;padding:18px}
h2 {font-size:22px;margin:2px 0 12px}
#status {color:#a7b9cc;font-size:13px;margin:0 0 10px}
#stage {position:relative; width:100%; background:#000; border-radius:12px;overflow:hidden}
canvas {display:block;width:100%;height:auto;aspect-ratio:16/9}
#help {font-size:13px;color:#b7c4d2;line-height:1.5;margin-top:12px}
</style></head><body><main>
<h2>C922 / Hailo-8 / Squat Live</h2>
<div id="status">연결 중... 영상과 JSON은 같은 frame_id로 전달됩니다.</div>
<div id="stage"><canvas id="canvas" width="800" height="450"></canvas></div>
<div id="help">카메라 영상은 R2130-12에서 전송하고, 스켈레톤·횟수·상태는 이 노트북의 브라우저 Canvas에서 직접 그립니다. 보드는 오버레이를 그리지 않습니다.</div>
</main>
<script>
'use strict';
const edges=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
const status=document.getElementById('status');
let latest=null, painting=false, lastPaintedFrame=-1;
const stream=new EventSource('/events');
stream.onopen=()=>{ status.textContent='스트림 연결됨 · 첫 프레임 수신 대기'; };
stream.onerror=()=>{ status.textContent='연결 재시도 중...'; };
stream.onmessage=event=>{
  try { latest=JSON.parse(event.data); if (!painting) paint(); }
  catch(err) { status.textContent='JSON 오류: '+err; }
};
function drawHud(p) {
  const W=canvas.width;
  ctx.fillStyle='rgba(0,0,0,0.70)'; ctx.fillRect(0,0,W,86);
  ctx.textBaseline='top'; ctx.textAlign='left';
  ctx.font='bold 25px system-ui'; ctx.fillStyle='#fff';
  ctx.fillText(`SQUAT  ${p.count}     Published  ${p.published}`,12,10);
  ctx.font='bold 15px system-ui';ctx.fillStyle='#facc15';
  ctx.fillText(`State: ${p.state}    Callback: ${Number(p.callback_fps).toFixed(1)} FPS    Frame: ${p.frame_id}    People: ${p.persons.length}`,12,51);
}
function drawSkeletons(p) {
  const sx=canvas.width/p.source_width, sy=canvas.height/p.source_height;
  for (const person of p.persons) {
    const selected = p.active_id==null || person.track_id===p.active_id;
    ctx.strokeStyle=selected?'#55ff72':'#fbbf24';
    ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=Math.max(2,canvas.width/400);
    for (const [a,b] of edges) {
      const x=person.joints[a], y=person.joints[b];
      if (!x||!y) continue;
      ctx.beginPath();ctx.moveTo(x[0]*sx,x[1]*sy);ctx.lineTo(y[0]*sx,y[1]*sy);ctx.stroke();
    }
    for (const [x,y] of person.joints) {
      if (!Number.isFinite(x)||!Number.isFinite(y)) continue;
      ctx.beginPath();ctx.arc(x*sx,y*sy,3.5,0,2*Math.PI);ctx.fill();
    }
  }
}
async function paint() {
  painting=true;
  while(latest) {
    const packet=latest; latest=null;
    if(packet.frame_id<=lastPaintedFrame) continue;
    const img=new Image();
    try {
      await new Promise((resolve,reject)=>{
        img.onload=resolve;img.onerror=()=>reject(new Error('JPEG decode failed'));
        img.src='data:image/jpeg;base64,'+packet.jpeg_base64;
      });
    } catch(error) { status.textContent=String(error);continue; }
    if(latest && latest.frame_id>packet.frame_id) continue; // skip stale frames
    if(canvas.width!==packet.video_width||canvas.height!==packet.video_height) {
      canvas.width=packet.video_width;canvas.height=packet.video_height;
    }
    ctx.drawImage(img,0,0,canvas.width,canvas.height);
    drawSkeletons(packet); drawHud(packet); lastPaintedFrame=packet.frame_id;
    status.textContent=`영상+JSON 동기화: frame ${packet.frame_id} · ${packet.video_width}×${packet.video_height} · 노트북 Canvas 렌더링`;
  }
  painting=false;
}
</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            data=PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path != '/events':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache, no-transform')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()
        seq=-1
        try:
            while not stop_signal.is_set():
                seq,data=encoded_slot.next_after(seq,timeout=3)
                if data is None:
                    self.wfile.write(b': keepalive\n\n')
                else:
                    self.wfile.write(b'data: '+data.encode('utf-8')+b'\n\n')
                self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):
            pass
        except Exception as exc:
            print('DASHBOARD CLIENT ERROR:',str(exc),flush=True)
    def log_message(self,*args):
        pass

if __name__ == '__main__':
    worker=threading.Thread(target=encoder_worker,daemon=True)
    worker.start()
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    server.daemon_threads=True
    th=threading.Thread(target=server.serve_forever,daemon=True)
    th.start()
    original.app_callback=pose_callback
    print(f'C922 raw video + pose JSON dashboard: http://127.0.0.1:{PORT}/',flush=True)
    print('RENDERING: laptop browser Canvas only; board does NOT draw pose/count overlays.',flush=True)
    print(f'Preview target {PREVIEW_FPS:g} fps, width <= {PREVIEW_WIDTH} (infer FPS is separate)',flush=True)
    try:
        original.main()
    finally:
        stop_signal.set()
        server.shutdown()
        server.server_close()
