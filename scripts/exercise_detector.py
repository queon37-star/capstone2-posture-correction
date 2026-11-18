"""운동 자동 판별기: 관절 좌표(COCO17)만 받아 squat / pushup / idle 을 판단한다.

- 입력: 한 프레임의 (17, 3) 배열 [x, y, confidence], 픽셀 좌표 (y는 아래로 증가)
        신뢰도 0.30 미만 관절은 NaN 이어도 되고 아니어도 된다 (여기서 다시 거른다).
- 출력: 확정된 판정 "squat" | "pushup" | "idle", 아직 확정 전이면 None
- 카메라·YOLO 에 의존하지 않으므로 PC(ultralytics)와 장치(Hailo) 어디서든 그대로 쓴다.
- 프레임 수가 아니라 '시간(초)' 기준으로 판단하므로 FPS가 달라도 동작이 같다.

v2: 정면 푸시업 오판 → '손목이 몸에서 가장 낮은 곳(바닥)'으로 엎드린 자세 구분
v4: 실제 관절 데이터(subject_001, 16개 영상) 분석 결과로 기준 변경
    - 엎드린 자세: '몸 전체의 가로폭 ÷ 세로높이'(가로로 넓으면 누운 것)를 추가
        실측: 푸시업 1.47~3.18 / 스쿼트 0.29~0.51 / 서서 하는 운동 0.43~0.74
        대각선 푸시업은 손목이 바닥으로 안 잡혀서 v2가 스쿼트로 오판했다.
    - 스쿼트: 골반 상하 이동을 '몸통 길이' 대신 '몸 세로높이'로 나눔
        실측: 스쿼트 0.28~0.33 / 벤트오버 로우 0.07~0.14
        정면에서 숙이면 몸통이 짧아 보여서 v2가 로우를 스쿼트로 오판했다.
v5: 다른 피험자(subject_002~004) 검증에서 얕게 앉는 스쿼트(subject_004 bad 3개)를 idle 로 오판
    - 원인: 골반 이동폭 0.15~0.16 으로 기준 0.20 미만. 그런데 로우도 최대 0.18 이라 기준만 낮추면 위험
    - 추가 기준 '골반 이동 ÷ 어깨 이동' (몸 전체가 같이 내려가는지)
        스쿼트: 몸이 통째로 내려감 → 0.43~0.95 (중앙값 0.54 이상)
        로우:   허리를 접어 어깨만 크게 움직임 → 대부분 0.2~0.5
    - 스쿼트 = 골반 이동 ≥ 0.20, 또는 (골반 이동 ≥ 0.10 이고 골반÷어깨 ≥ 0.50)
    - 결과: subject_001~004, 70개 영상 70/70 (v4 는 67/70), 잘못 확정된 시간 4.5% → 0.9%
    - 주의: 0.50 은 얕은 스쿼트(0.45~0.53)와 로우 최대값 사이라 여유가 작다. 0.55 부터 다시 틀림
"""
from collections import deque

import numpy as np

# COCO17 관절 번호 (시원 님 stream_squat_core.py 와 동일)
NOSE = 0
LS, RS, LW, RW, LH, RH, LA, RA = 5, 6, 9, 10, 11, 12, 15, 16
CONF = 0.30


def _point(kp, i):
    x, y, c = kp[i]
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)) or c < CONF:
        return None
    return np.array([x, y], dtype=float)


def _center(kp, a, b):
    p, q = _point(kp, a), _point(kp, b)
    if p is None and q is None:
        return None
    if p is None:
        return q
    if q is None:
        return p
    return (p + q) / 2


def measure(kp):
    """한 프레임에서 판별에 쓰는 값 계산. 필요한 관절이 없으면 None."""
    kp = np.asarray(kp, dtype=float)
    if kp.shape != (17, 3):
        raise ValueError("Expected (17, 3) COCO17 keypoints")
    sh, hip = _center(kp, LS, RS), _center(kp, LH, RH)
    if sh is None or hip is None:
        return None
    pts = [p for p in (_point(kp, i) for i in range(17)) if p is not None]
    ys = np.array([p[1] for p in pts])
    height = float(ys.max() - ys.min())            # 화면상 몸 전체의 세로 크기
    if height < 20:
        return None
    wrist, ankle = _center(kp, LW, RW), _center(kp, LA, RA)
    # 손목이 몸의 가장 아래쪽(바닥) 근처에 있으면 1 = '손을 바닥에 짚은 자세'
    #  - 손목 y 가 발목 y 와 같거나 더 아래 (측면 푸시업: 손·발 모두 바닥)
    #  - 발목이 안 보이거나 더 위에 보임 (정면 푸시업: 발이 멀어서 위쪽에 보임)
    hands_down = None
    if wrist is not None:
        lowest = ys.max()
        wrist_is_lowest = (lowest - wrist[1]) <= 0.15 * height
        if ankle is not None:
            hands_down = wrist_is_lowest and (wrist[1] >= ankle[1] - 0.15 * height)
        else:
            hands_down = wrist_is_lowest
    xs = np.array([p[0] for p in pts])
    width = float(xs.max() - xs.min())             # 화면상 몸 전체의 가로폭
    return {"hands_down": hands_down, "height": height, "width_ratio": width / height,
            "hip_y": float(hip[1]), "sh_y": float(sh[1])}


def _spread(values):
    return float(np.percentile(values, 90) - np.percentile(values, 10))


class ExerciseDetector:
    def __init__(self,
                 window_sec=2.0,         # 움직임을 볼 시간 범위
                 confirm_sec=1.5,        # 같은 판정이 이만큼 이어져야 확정
                 hands_down_ratio=0.6,   # 창 안에서 '손이 바닥' 프레임 비율이 이 이상이면 엎드린 자세
                 lying_ratio=1.0,        # 몸 가로폭÷세로높이가 이 이상이면 엎드린 자세 (실측 경계: 서서 0.78 / 푸시업 1.18)
                 squat_motion=0.20,      # 골반 상하 이동폭이 이 이상이면 바로 스쿼트 (몸 세로높이 단위)
                 squat_min_motion=0.10,  # 얕은 스쿼트: 골반 이동폭이 이 이상이고
                 squat_sync=0.50,        #   골반 이동 ÷ 어깨 이동이 이 이상이면 스쿼트 (로우는 어깨만 크게 움직임)
                 pushup_motion=0.08,     # 어깨 상하 이동폭 (화면상 몸 세로 크기 단위)
                 min_valid_ratio=0.5):   # 창 안에서 관절이 보인 프레임 비율 최소값
        self.window_sec = window_sec
        self.confirm_sec = confirm_sec
        self.hands_down_ratio = hands_down_ratio
        self.lying_ratio = lying_ratio
        self.squat_motion = squat_motion
        self.squat_min_motion = squat_min_motion
        self.squat_sync = squat_sync
        self.pushup_motion = pushup_motion
        self.min_valid_ratio = min_valid_ratio
        self.reset()

    def reset(self):
        self.buf = deque()            # (t, measure or None)
        self.candidate = None         # 확정 대기 중인 판정
        self.candidate_since = None
        self.confirmed = None         # 확정된 판정
        self.last_raw = None
        self.last_info = {}           # 디버깅용: 판단에 쓴 수치

    def raw_label(self):
        """현재 창만 보고 내린 즉석 판정 (흔들릴 수 있음)."""
        valid = [m for _, m in self.buf if m is not None]
        self.last_info = {}
        if not self.buf or len(valid) < self.min_valid_ratio * len(self.buf) or len(valid) < 5:
            return None                                   # 사람이 잘 안 보임 → 판단 보류
        hd = [m["hands_down"] for m in valid if m["hands_down"] is not None]
        hands_ratio = float(np.mean(hd)) if hd else 0.0
        height = float(np.median([m["height"] for m in valid]))
        width_ratio = float(np.median([m["width_ratio"] for m in valid]))
        hip_motion = _spread([m["hip_y"] for m in valid]) / height
        sh_motion = _spread([m["sh_y"] for m in valid]) / height
        hip_sync = hip_motion / max(sh_motion, 1e-6)      # 1에 가까우면 몸 전체가 같이 오르내림
        self.last_info = {"hands_down_ratio": round(hands_ratio, 2),
                          "width_ratio": round(width_ratio, 2),
                          "hip_motion": round(hip_motion, 3),
                          "shoulder_motion": round(sh_motion, 3),
                          "hip_sync": round(hip_sync, 2)}
        lying = width_ratio >= self.lying_ratio or hands_ratio >= self.hands_down_ratio
        if lying:                                         # 몸이 가로로 넓거나 손을 바닥에 짚음 → 엎드린 자세
            return "pushup" if sh_motion >= self.pushup_motion else "idle"
        if hip_motion >= self.squat_motion:               # 깊게 앉음 → 스쿼트
            return "squat"
        if hip_motion >= self.squat_min_motion and hip_sync >= self.squat_sync:
            return "squat"                                # 얕게 앉지만 몸 전체가 같이 내려감 → 스쿼트
        return "idle"

    def update(self, kp, t):
        """매 프레임 호출. t = 영상 시작부터의 시간(초). 확정된 판정을 돌려준다."""
        self.buf.append((t, measure(kp)))
        while self.buf and self.buf[0][0] < t - self.window_sec:
            self.buf.popleft()
        raw = self.raw_label()
        self.last_raw = raw
        if raw is None:
            return self.confirmed
        if raw != self.candidate:
            self.candidate, self.candidate_since = raw, t
        elif t - self.candidate_since >= self.confirm_sec and raw != self.confirmed:
            self.confirmed = raw
        return self.confirmed
