"""운동 자동 판별 + 스쿼트 카운팅/분류 + 서버 전송 (실시간 카메라 / 영상 파일 공용)

run_squat_video_stream_tunable.py 를 바탕으로 아래를 추가한 실행 파일이다.
기존 파일(카운터, 버퍼, 분류기)은 수정하지 않고 import 해서 그대로 쓴다.

  1) 수행자 선택: 수동 Anchor 대신 '처음엔 가장 큰 사람 → 이후 직전 박스와 IoU 최대'
     (1초 이상 놓치면 다시 가장 큰 사람)  → 카메라 입력에서도 설정 파일 없이 동작
  2) 운동 판별기(exercise_detector.py): 매 프레임 squat / pushup / idle 판정
  3) 세션 관리
       idle → 운동 확정      : session_start 전송 (squat 이면 카운터·버퍼 새로 생성)
       운동 → idle 이 idle_end_sec 이상 지속 / 사람이 lost_sec 이상 안 보임 : session_end 전송
       squat ↔ pushup 직접 전환 : 이전 세션 session_end → 새 세션 session_start
  4) squat 확정 시 최근 replay_sec 초 프레임을 새 카운터에 다시 넣음
       (판별기가 확정까지 2~4초 걸리므로, 그 사이 첫 반복을 놓치지 않기 위함)
  5) pushup 은 카운터가 아직 없으므로 session_start / session_end 만 전송
  6) 이벤트를 JSONL 파일에 기록 + 서버 /internal/update 로 POST
       (별도 스레드에서 전송, 실패해도 인식은 계속)

실행 예 (저장소 루트에서):
  영상 파일:  python scripts/run_auto_stream.py --source test.mp4 --device cpu --no-post
  USB 카메라: python scripts/run_auto_stream.py --source 0 --show
  서버 주소 변경: --server http://127.0.0.1:8000/internal/update

※ 장치(Hailo)에서는 estimate_pose() 함수 하나만 Hailo 추론으로 바꾸면 된다.
   반환 형식: boxes (N,4) xyxy 픽셀, scores (N,), keypoints (N,17,3) [x, y, conf] 픽셀
"""
import argparse
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from exercise_detector import ExerciseDetector
from extract_all_squats_tracked import box_iou
from run_squat_video_stream import classify, absent
from stream_squat_core import SquatCounter
from squat_rep_buffer import RepBoundaryBuffer

EXERCISES = ('squat', 'pushup')


# ---------------------------------------------------------------- 포즈 추론
class PoseEstimator:
    """PC(ultralytics) 용. 장치에서는 같은 반환 형식의 Hailo 버전으로 교체."""

    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device

    def __call__(self, frame):
        r = self.model.predict(frame, device=self.device, imgsz=640, conf=.25, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
        return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                r.keypoints.data.cpu().numpy().astype(np.float32))


# ---------------------------------------------------------------- 수행자 선택
class PersonTracker:
    """처음엔 가장 큰 사람, 이후 직전 박스와 IoU 최대인 사람. 오래 놓치면 다시 가장 큰 사람."""

    def __init__(self, reacquire_sec=1.0, min_iou=0.1):
        self.reacquire_sec = reacquire_sec
        self.min_iou = min_iou
        self.prev_box = None
        self.last_seen = -1e9

    def select(self, boxes, t):
        if len(boxes) == 0:
            return None
        if self.prev_box is not None and t - self.last_seen <= self.reacquire_sec:
            ious = np.array([box_iou(b, self.prev_box) for b in boxes])
            idx = int(ious.argmax())
            if ious[idx] < self.min_iou:
                return None                      # 이어지는 사람이 없음 → 이번 프레임은 놓친 것으로
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = int(areas.argmax())
        self.prev_box = boxes[idx].copy()
        self.last_seen = t
        return idx


# ---------------------------------------------------------------- 서버 전송
class EventSender:
    """이벤트를 JSONL 에 기록하고, 별도 스레드에서 서버로 POST (인식 루프를 막지 않음)."""

    SERVER_FIELDS = ('type', 'exercise', 'count', 'posture', 'issues', 'score')

    def __init__(self, out_path, server_url=None, timeout=1.0):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = out_path.open('a', encoding='utf-8')
        self.server_url = server_url
        self.timeout = timeout
        self.q = queue.Queue(maxsize=200)
        if server_url:
            import requests
            self.session = requests.Session()
            threading.Thread(target=self._worker, daemon=True).start()

    def send(self, message):
        self.file.write(json.dumps(message, ensure_ascii=False) + '\n')
        self.file.flush()
        print('EVENT', json.dumps({k: message[k] for k in self.SERVER_FIELDS if k in message},
                                  ensure_ascii=False), flush=True)
        if self.server_url:
            try:
                self.q.put_nowait(message)
            except queue.Full:
                print('WARN 서버 전송 대기열이 가득 차서 이벤트 1개를 버림', flush=True)

    def _worker(self):
        while True:
            message = self.q.get()
            try:
                r = self.session.post(self.server_url, json=message, timeout=self.timeout)
                if r.status_code != 200:
                    print(f'WARN 서버 응답 {r.status_code}: {r.text[:200]}', flush=True)
            except Exception as exc:                     # 서버가 꺼져 있어도 인식은 계속
                print(f'WARN 서버 전송 실패: {exc}', flush=True)

    def close(self, wait_sec=2.0):
        end = time.time() + wait_sec
        while self.server_url and not self.q.empty() and time.time() < end:
            time.sleep(0.05)
        self.file.close()


# ---------------------------------------------------------------- 세션 관리
class SessionManager:
    def __init__(self, args, fps, artifact, sender):
        self.args, self.fps, self.artifact, self.sender = args, fps, artifact, sender
        self.exercise = None          # 현재 세션 운동 (None = 세션 없음)
        self.idle_since = None
        self.counter = self.assembler = None
        self.rep_count = 0
        self.recent = deque(maxlen=int(round(args.replay_sec * fps)))   # (fid, kp)

    # --- 이벤트 만들기
    def _event(self, type_, **extra):
        msg = {'type': type_, 'exercise': self.exercise, 'count': self.rep_count,
               'posture': 'unknown', 'issues': [], 'score': None}
        msg.update(extra)
        self.sender.send(msg)

    def _publish_rep(self, rep):
        self.rep_count += 1
        try:
            pred, prob = classify(rep.keypoints, self.artifact)
            extra = {'posture': 'good' if pred == 1 else 'bad',
                     'probability_correct_uncalibrated': round(prob, 4)}
        except Exception as exc:
            extra = {'posture': 'unknown', 'classification_error': str(exc)}
        extra.update({'start_frame': rep.start_frame, 'end_frame': rep.end_frame,
                      'post_context_complete': rep.post_context_complete,
                      'peak_hip_depth_torso_units': round(rep.peak_depth, 3),
                      'min_projected_knee_angle': round(rep.min_knee_angle, 1)})
        self._event('rep', **extra)

    def _push_counter(self, fid, kp):
        rep_event = self.counter.push(kp, frame_id=fid)
        for rep in self.assembler.push(fid, kp, rep_event):
            self._publish_rep(rep)

    # --- 세션 시작/종료
    def start(self, exercise, t):
        self.exercise, self.idle_since, self.rep_count = exercise, None, 0
        print(f'[{t:7.1f}s] SESSION START {exercise}', flush=True)
        self._event('session_start')
        if exercise == 'squat':
            a = self.args
            self.counter = SquatCounter(fps=self.fps, bottom_depth=a.bottom_depth, bottom_knee=a.bottom_knee)
            self.assembler = RepBoundaryBuffer(fps=self.fps, pre_frames=a.pre_frames, post_frames=a.post_frames)
            for fid, kp in list(self.recent):           # 확정 전 최근 프레임 다시 넣기 (첫 반복 보존)
                self._push_counter(fid, kp)

    def end(self, t, reason):
        if self.exercise is None:
            return
        if self.assembler is not None:
            for rep in self.assembler.flush():
                self._publish_rep(rep)
        print(f'[{t:7.1f}s] SESSION END {self.exercise} ({reason}), reps={self.rep_count}', flush=True)
        self._event('session_end')
        self.exercise = self.idle_since = self.counter = self.assembler = None

    # --- 매 프레임
    def update(self, fid, kp, t, label, person_lost_sec):
        self.recent.append((fid, kp))
        if self.exercise == 'squat':
            self._push_counter(fid, kp)

        if self.exercise is not None and person_lost_sec >= self.args.lost_sec:
            self.end(t, 'person lost')
        elif label in EXERCISES and label != self.exercise:
            self.end(t, f'switch to {label}')
            self.start(label, t)
        elif label == self.exercise:
            self.idle_since = None
        elif self.exercise is not None:                 # 확정 판정이 idle (또는 아직 없음)
            if self.idle_since is None:
                self.idle_since = t
            elif t - self.idle_since >= self.args.idle_end_sec:
                self.end(t, 'idle')


# ---------------------------------------------------------------- 메인
def open_source(source, width, height, fps):
    is_camera = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_camera else source)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {source}')
    if is_camera:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap, is_camera


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='0', help='카메라 번호(0, 1 …) 또는 영상 파일 경로')
    p.add_argument('--model', type=Path, default=Path('models/squat_plus_asymmetry180_crop10_all9.joblib'))
    p.add_argument('--yolo', default='yolov8m-pose.pt')
    p.add_argument('--device', default='0', help='GPU 번호 또는 cpu')
    p.add_argument('--fps', type=float, default=0, help='카운터에 줄 FPS (0이면 영상/카메라 값 사용)')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--bottom-depth', type=float, default=0.20)
    p.add_argument('--bottom-knee', type=float, default=120.0)
    p.add_argument('--pre-frames', type=int, default=16)
    p.add_argument('--post-frames', type=int, default=22)
    p.add_argument('--replay-sec', type=float, default=6.0, help='squat 확정 시 카운터에 다시 넣을 최근 시간')
    p.add_argument('--idle-end-sec', type=float, default=3.0, help='확정 idle 이 이만큼 이어지면 session_end')
    p.add_argument('--lost-sec', type=float, default=3.0, help='사람이 이만큼 안 보이면 session_end')
    p.add_argument('--server', default='http://127.0.0.1:8000/internal/update')
    p.add_argument('--no-post', action='store_true', help='서버 전송 끄기 (JSONL 기록만)')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--show', action='store_true', help='화면에 뼈대·판정 표시 (q: 종료)')
    p.add_argument('--max-frames', type=int, default=0)
    args = p.parse_args()

    artifact = joblib.load(args.model)  # Only load own trusted model file.
    if artifact.get('feature_variant') != 'PlusAsymmetry180':
        raise ValueError('Expected PlusAsymmetry180 classifier artifact')
    cap, is_camera = open_source(args.source, args.width, args.height, args.fps or 30)
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = args.output or Path(f'results/auto_stream/{time.strftime("%Y%m%d-%H%M%S")}_events.jsonl')
    sender = EventSender(out, None if args.no_post else args.server)
    estimate_pose = PoseEstimator(args.yolo, args.device)
    tracker = PersonTracker()
    detector = ExerciseDetector()
    session = SessionManager(args, fps, artifact, sender)
    print(f'source={args.source} fps={fps:.1f} output={out} server={"off" if args.no_post else args.server}',
          flush=True)

    t0 = time.perf_counter()
    n = found = 0
    t = 0.0
    try:
        while args.max_frames <= 0 or n < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            # 카메라는 실제 경과 시간, 영상 파일은 프레임 번호 기준 시간
            t = time.perf_counter() - t0 if is_camera else n / fps
            boxes, scores, kps = estimate_pose(frame)
            kp = absent()
            idx = tracker.select(boxes, t)
            if idx is not None:
                kp = kps[idx].astype(np.float32).copy()
                kp[kp[:, 2] < .3, :2] = np.nan
                found += 1
            lost_sec = t - tracker.last_seen
            if lost_sec >= args.lost_sec and (detector.confirmed is not None or detector.buf):
                detector.reset()                 # 사람이 오래 안 보이면 이전 판정을 버리고 처음부터
            label = detector.update(kp, t) if lost_sec < args.lost_sec else None
            session.update(n, kp, t, label, person_lost_sec=lost_sec)
            n += 1

            if n % 150 == 0:
                el = max(time.perf_counter() - t0, 1e-6)
                print(f'frame={n} label={label} raw={detector.last_raw} session={session.exercise} '
                      f'reps={session.rep_count} pose_found={found}/{n} processing_fps={n / el:.1f}',
                      flush=True)
            if args.show:
                for x, y, c in kp:
                    if np.isfinite(x) and c >= .3:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (25, 25, 25), -1)
                cv2.putText(frame, f'{label or "..."}  (raw {detector.last_raw})  reps {session.rep_count}',
                            (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                cv2.imshow('auto stream (q: quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print('중지 요청 (Ctrl+C)', flush=True)
    finally:
        session.end(t, 'stream end')
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        sender.close()
        el = max(time.perf_counter() - t0, 1e-6)
        print(f'DONE: {n} frames, pose_found={found}/{n or 1}, processing_fps={n / el:.1f}, output={out}')


if __name__ == '__main__':
    main()
