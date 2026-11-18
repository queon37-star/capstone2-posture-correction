"""C922 live pose / squat web preview, without modifying the original counter.

Run from ~/capstone2/pose_training as:
  python -u scripts/run_squat_pose_web.py --input /dev/video0 --arch hailo8 --hef-path ...
The original script run_squat_hailo_pose.py must be in the same scripts directory.
"""

import time
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import hailo
from hailo_apps_infra.hailo_rpi_common import get_numpy_from_buffer

import run_squat_hailo_pose as original

# COCO17 joint links, zero-based indices
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
         (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

class LatestView:
    def __init__(self):
        self.cv = threading.Condition()
        self.payload = None
        self.seq = 0

    def update(self, frame, people, status, fmt):
        with self.cv:
            self.seq += 1
            self.payload = (self.seq, frame, people, status, fmt)
            self.cv.notify_all()

    def wait_next(self, old_seq):
        with self.cv:
            self.cv.wait_for(lambda: self.seq != old_seq, timeout=1.0)
            return self.payload

view = LatestView()
callback_times = deque(maxlen=120)
original_callback = original.app_callback
error_shown = False


def extract_people(buffer, width, height):
    people = []
    detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)
    for det in detections:
        if det.get_label() != 'person':
            continue
        marks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not marks:
            continue
        bbox = det.get_bbox()
        coords = []
        points = marks[0].get_points()
        for i in range(min(17, len(points))):
            point = points[i]
            x = int((point.x() * bbox.width() + bbox.xmin()) * width)
            y = int((point.y() * bbox.height() + bbox.ymin()) * height)
            coords.append((x, y))
        ids = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        track_id = ids[0].get_id() if ids else None
        people.append((track_id, coords))
    return people


def pose_callback(pad, info, user_data):
    global error_shown
    # Keep the original pose-to-counter handling exactly as it was.
    result = original_callback(pad, info, user_data)

    now = time.perf_counter()
    callback_times.append(now)
    # Copy only 1 in 3 processed frames; browser rendering is on another thread.
    if user_data.get_count() % 3:
        return result

    try:
        buffer = info.get_buffer()
        if buffer is None:
            return result
        fmt, width, height = original.get_caps_from_pad(pad)
        if fmt is None or width is None or height is None:
            return result
        frame = get_numpy_from_buffer(buffer, fmt, width, height)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f'Unsupported frame shape: {None if frame is None else frame.shape}, format={fmt}')

        people = extract_people(buffer, width, height)
        bridge = user_data.bridge
        active_id = bridge.selector.active_track_id if bridge.selector else None
        fps = 0.0
        if len(callback_times) > 1:
            span = callback_times[-1] - callback_times[0]
            fps = (len(callback_times) - 1) / span if span > 0 else 0.0
        status = {
            'count': bridge.pipeline.counter.count,
            'published': bridge.pipeline.event_count,
            'state': bridge.pipeline.counter.state,
            'fps': fps,
            'frame_id': user_data.get_count() - 1,
            'active_id': active_id,
        }
        # gst buffer may be returned to the pipeline after callback exits.
        view.update(np.ascontiguousarray(frame.copy()), people, status, str(fmt).upper())
    except Exception as exc:
        if not error_shown:
            print(f'WEB PREVIEW ERROR: {type(exc).__name__}: {exc}', flush=True)
            error_shown = True
    return result


def make_jpeg(payload):
    _, frame, people, status, fmt = payload
    if fmt == 'RGB':
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif fmt == 'BGR':
        img = frame.copy()
    else:
        raise ValueError(f'Unexpected GStreamer pixel format: {fmt}')

    for track_id, coords in people:
        selected = status['active_id'] is None or track_id == status['active_id']
        color = (20, 245, 50) if selected else (0, 210, 255)
        for a, b in EDGES:
            if len(coords) > max(a, b):
                cv2.line(img, coords[a], coords[b], color, 2)
        for x, y in coords:
            cv2.circle(img, (x, y), 3, color, -1)
        if coords:
            cv2.putText(img, f'ID: {track_id}', coords[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, min(h, 82)), (22, 22, 22), -1)
    cv2.putText(img,
                f'Squat: {status["count"]}   Published: {status["published"]}   State: {status["state"]}',
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img,
                f'Callback FPS: {status["fps"]:.1f}   Frame: {status["frame_id"]}   People: {len(people)}',
                (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 240, 240), 2)
    # Limit preview bandwidth; inference works at its original resolution.
    if w > 800:
        img = cv2.resize(img, (800, round(h * 800 / w)), interpolation=cv2.INTER_AREA)
    ok, output = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise RuntimeError('JPEG encode failed')
    return output.tobytes()


class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            page = (b'<!doctype html><html><meta name="viewport" content="width=device-width">'
                    b'<title>C922 Hailo Pose</title><body style="background:#171717;color:white;text-align:center;font-family:sans-serif">'
                    b'<h2>C922 / Hailo-8 / Squat Live</h2><img style="max-width:98%;height:auto" src="/stream">'
                    b'</body></html>')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path != '/stream':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        last_seq = -1
        try:
            while True:
                payload = view.wait_next(last_seq)
                if payload is None or payload[0] == last_seq:
                    continue
                last_seq = payload[0]
                jpg = make_jpeg(payload)
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                 + f'Content-Length: {len(jpg)}\r\n\r\n'.encode()
                                 + jpg + b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(f'WEB CLIENT ERROR: {type(exc).__name__}: {exc}', flush=True)

    def log_message(self, *_args):
        return


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8765), WebHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print('C922 Pose dashboard: http://127.0.0.1:8765/ (SSH tunnel required)', flush=True)
    original.app_callback = pose_callback
    try:
        original.main()
    finally:
        server.shutdown()
        server.server_close()
