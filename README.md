# 경량 AI 실시간 운동 자세 교정 시스템

> 융합캡스톤디자인2 · 4조 · AI 분석 파트 개발 문서  
> **기준일: 2026-10-14** · 현재 저장소는 **개발 PC에서 검증한 스쿼트 AI 프로토타입** 중심

## 0. 통합 시 참고 사항

- **프로젝트 목적:** 카메라 영상에서 운동 수행자의 관절을 추출해 스쿼트 횟수와 반복별 `Good`/`Bad`를 판단하고, 결과를 앱에 전달하는 시스템 개발.
- **AI 분석:** YOLOv8m-Pose(COCO17) → 수행자 추적 → SquatCounter(FSM) → 반복 구간 보완 → 180개 Feature → Logistic Regression → `rep` 이벤트.
  - **카운팅:** 골반 이동량·무릎 각도를 이용하는 규칙 기반 FSM.
  - **Good/Bad 분류:** 완료된 **스쿼트 1회 전체**의 특징으로 판단하는 이진 분류기. 프레임별 자세 오류 경고 모델은 아님.
- **개발 상태 구분:** PC 저장 영상에서 AI 분석·평가 완료 / R2130-12에서 Hailo-8 Pose 샘플 추론 확인 / **실제 카메라→AI→서버→앱 통합 미완료**.
- **서버 측 구현:** Raspberry Pi의 FastAPI + ngrok, 외부 WebSocket 접속, JSON 형식 검증, `systemd` 자동 실행·재시작. 현재는 `random_sender`의 **가상 AI 데이터**로 앱 연동을 확인함.
- **앱 측 구현:** Expo/React Native, `wss` 연결·재연결, `rep` 횟수/정오 집계, `posture` 피드백 표시, 운동 종료 시 **휴대폰 SQLite 로컬 저장**.
- **핵심 연동 사항:** 현재 AI 프로그램은 `JSONL` 파일에 결과 기록. **FastAPI로 실제 POST 전송하는 코드와 실제 Pose 입력 연결은 추가 구현 필요**. AI의 `issues=[]`, `score=null`은 현재 모델의 정상적인 출력 형식.
- **현재 AI 실행 설정:** `models/squat_plus_asymmetry180_crop10_all9.joblib`, `bottom_depth=0.20`, 반복 앞 **16프레임**·뒤 **22프레임** 추가.

## 1. 전체 시스템과 현재 개발 범위

### 1-1. 프로젝트 전체 구성(목표와 현재 구현 구분)

```text
[현재 PC에서 검증한 AI 처리 / 보드 통합 목표]
카메라 또는 저장 영상
    ↓
reComputer AI R2130-12 + Hailo-8    ← 실제 카메라 기반 전체 연결은 진행 필요
    ├─ YOLOv8m-Pose / COCO17 관절 추정
    ├─ 운동 수행자 선택·추적
    ├─ SquatCounter / 반복 구간 보완
    ├─ 180개 특징 / Logistic Regression
    └─ AI 반복 이벤트(JSON)             ← 현재 PC 구현은 JSONL 기록
           ↓ HTTP POST 연동 어댑터 개발 필요
[서버: 현재 Raspberry Pi에서 별도 동작]
    FastAPI → JSON 검사 → 이벤트 기록 / WebSocket 전송
    ngrok 고정 도메인 + systemd 서비스
           ↓ WSS(/ws), 인터넷 접속
[앱: Expo / React Native]
    실시간 반복 횟수·Good/Bad·자세 경고 표시
    └─ 운동 종료 시 휴대폰 SQLite에 기록 저장
```

| 담당 영역 | 중간 보고서 기준 확인된 구현 | AI와 연결할 때 필요한 작업 |
|---|---|---|
| **AI 분석** | PC 저장 영상에서 Pose → 추적 → 카운팅 → Good/Bad → JSONL 출력; R2130-12에서 Hailo Pose 샘플 작동 확인 | 라이브 카메라·Hailo 출력 연결, 서버에 이벤트 POST |
| **서버(손준표)** | Raspberry Pi + FastAPI + ngrok 고정 주소, 외부 WSS 수신 확인, JSON 스키마 검사, `systemd` 자동 실행·재시작, 테스트 이벤트 송신 | 실제 AI 이벤트 수신·검증 및 보드/서버 배치 확정 |
| **앱(임태빈)** | Expo 앱의 WebSocket 수신·재연결, 스쿼트·푸시업 선택 UI, 운동 종료 시 SQLite 로컬 저장 | 실제 AI 출력 검증, 오류 코드·운동 필터링·세션 정책 확정 |

- **현재 서버–앱 연동은 실제 AI 판별이 아닌 `random_sender` 등 시뮬레이션 이벤트로 확인한 결과**임.
- 서버와 앱의 저장소/실행 환경은 이 **AI 코드 저장소와 별도**이므로 본 저장소의 Python 스크립트만 실행해서 앱이 자동 연결되지는 않음.
- **서버 배치:** 중간 보고서 시점에는 별도 Raspberry Pi에서 운영. R2130-12로 이전하여 AI·서버를 함께 구동하는 단계는 향후 통합 작업.

### 1-2. AI 처리 과정(현재 PC 구현)

```text
원본 MP4 (별도 반복 구간 분할 없음)
   → 프레임별 YOLOv8m-Pose 추론
   → 여러 검출 인물 중 수행자 1명 선택
   → 수행자의 Keypoint 배열 [T, 17, 3] 누적
   → SquatCounter: 스쿼트 1회 완료 검출
   → RepBoundaryBuffer: -16 / +22프레임 보완
   → 16개 기본 신호 × 10개 통계 = 160개 특징
     + 4개 비대칭 신호 × 5개 통계 = 20개 특징
   → 180차원 벡터 → Logistic Regression
   → count / posture / 구간 정보 등을 JSONL에 저장
```

- **온라인(순차) 분석 방식:** 원본 영상의 프레임을 순서대로 처리하며 반복 구간을 자동 검출. 추론 과정에서는 정답 구간(`Segmentation.csv`)을 사용하지 않음.
- **주의:** PC의 저장 영상 처리 방식이 구현되었다는 의미이며, **라이브 USB 카메라와 R2130-12에서의 실시간 동작이 검증되었다는 의미는 아님**.

## 2. 데이터셋과 Keypoint 형식

### 2-1. 학습·평가 데이터

| 항목 | 내용 |
|---|---|
| 데이터셋 | REHAB24-6 |
| 운동 종류 | Exercise 6: Squat |
| 실제 사용한 영상 | Camera17, 30 FPS, 9개 영상 |
| 피험자 | 9명 |
| 스쿼트 반복 | 총 **195회** |
| 정답 라벨 | `1 = Good(정상)` 134회 / `0 = Bad(비정상)` 61회 |
| 사용하지 않은 시점 | 동일 동작을 다른 카메라에서 찍은 Camera18(향후 평가 후보) |

- 학습용 반복 시퀀스 추출 시 `Segmentation.csv`의 구간 및 정오 라벨을 사용.
- **실제 스트리밍 추론 시에는 해당 정답 구간·라벨을 사용하지 않음.** 영상 추론이 끝난 후 성능 평가 목적으로만 정답과 비교.
- 원본 데이터, 영상, 처리된 Keypoint(`*.npz`)는 Git 저장소에 포함하지 않음.

### 2-2. YOLOv8m-Pose 출력

- **관절 포맷:** COCO17.
- **한 프레임의 한 사람:** `(17, 3)` 크기의 배열.
  - `keypoints[i] = [x, y, confidence]`.
  - `x, y`: 영상 내 픽셀 좌표. 이미지 좌측 상단이 원점이며 **아래로 갈수록 y 증가**.
  - `confidence`: 각 관절 검출 신뢰도.
- **한 번의 스쿼트:** `(T, 17, 3)` 배열. `T`는 그 반복에 포함된 프레임 수.
- **관절 신뢰도 기준:** 현재 특징 계산에서는 `confidence < 0.30`인 좌표를 결측값(`NaN`)으로 처리.
- **핵심 관절:** 어깨(5, 6), 골반(11, 12), 무릎(13, 14), 발목(15, 16). **모델은 17개를 출력하지만 현재 180개 특징 대부분은 이 8개 관절로 계산**.

| 인덱스 | 관절 | 인덱스 | 관절 |
|---:|---|---:|---|
| 0 | 코 | 1 | 왼쪽 눈 |
| 2 | 오른쪽 눈 | 3 | 왼쪽 귀 |
| 4 | 오른쪽 귀 | 5 | 왼쪽 어깨 |
| 6 | 오른쪽 어깨 | 7 | 왼쪽 팔꿈치 |
| 8 | 오른쪽 팔꿈치 | 9 | 왼쪽 손목 |
| 10 | 오른쪽 손목 | 11 | 왼쪽 골반 |
| 12 | 오른쪽 골반 | 13 | 왼쪽 무릎 |
| 14 | 오른쪽 무릎 | 15 | 왼쪽 발목 |
| 16 | 오른쪽 발목 |  |  |

### 2-3. 여러 사람 중 운동 수행자 추적

- 현재 방식: **영상별 수동 지정 Anchor(기준 Bounding Box)** + 이전 프레임의 Bounding Box 비교.
- 후보 선택 기준: Anchor와의 거리, 이전 선택 박스와의 거리 및 IoU(겹침), 검출 신뢰도.
- 검출 신뢰도가 가장 높은 사람을 무조건 선택하지 않음.
- 추적 대상이 검출되지 않으면 해당 프레임 Keypoint를 결측값으로 처리.
- **제약:** REHAB24-6의 각 영상에서 수행자를 사전에 수동 지정함. 실제 사용자 환경에서 자동 대상 선택·재식별(Re-ID)은 미구현.
- **코드:** `scripts/select_subjects.py`, `scripts/extract_all_squats_tracked.py`, `configs/rehab24_squat_camera17_subjects.json`.

## 3. 스쿼트 카운팅: SquatCounter(FSM)

### 3-1. 기본 원리

- **학습 모델이 아닌 규칙 기반 FSM(Finite State Machine)** 사용.
- 상태 순서: `CALIBRATING → READY → DESCENDING → BOTTOM → ASCENDING → READY`.
- 사용 신호:
  - **골반 높이:** 좌우 골반 중심의 `y` 좌표.
  - **무릎 각도:** `골반–무릎–발목`의 2D 관절 각도(좌우 유효값 평균).
  - **몸통 길이:** 어깨 중심과 골반 중심 사이 거리. 신체 크기·카메라 거리 영향을 줄이기 위한 정규화 기준.
- 초기 직립 자세에서 약 10개 유효 프레임으로 기준 골반 높이·몸통 길이를 보정.
- 골반 이동 깊이(몸통 길이 단위):

  `depth = (현재 골반 중심 y - 초기 직립 골반 중심 y) / 몸통 길이`

### 3-2. 최신 실행 설정에서 사용하는 상태 전이 조건

| 상태 변화 | 조건 | 연속 프레임 등 |
|---|---|---|
| `READY → DESCENDING` | `depth ≥ 0.16` **또는** 무릎 각도 `≤ 145°` | 3프레임 |
| `DESCENDING → BOTTOM` | `depth ≥ 0.20` **또는** 무릎 각도 `≤ 120°` | 3프레임 |
| `BOTTOM → ASCENDING` | 최고 깊이 대비 `depth` 0.18 이상 감소 **또는** 최소 무릎 각도 대비 15° 이상 증가 | 변화 감지 시 |
| `ASCENDING → READY` | `depth ≤ 0.13` **그리고** 무릎 각도 `≥ 150°` | 4프레임 |

- 관절 검출이 누락된 경우에는 일부 각도 조건을 생략하는 예외 처리 존재.
- 최근 5프레임의 유효 측정값 중앙값을 사용해 좌표 흔들림 완화.
- 반복 길이 최소 15프레임. 너무 길거나 추적이 장시간 끊긴 반복은 초기화 가능.
- **카운팅과 판정은 별개:** `Bad` 자세라도 유효한 반복으로 검출되면 `count +1`.
- **이전 실험 기본값 `bottom_depth=0.45`에 주의:** 현재 평가에 사용한 최종 실행 설정은 `0.20`이며 `scripts/run_squat_latest.sh`에서 명시적으로 지정.
- **코드:** `scripts/stream_squat_core.py`(FSM), `scripts/run_squat_video_stream_tunable.py`(임계값 전달).

### 3-3. 반복 구간 보정(RepBoundaryBuffer)

- 자동 카운터가 검출한 반복 구간은 실제 움직임의 시작·끝보다 짧을 수 있음.
- 분류 특징에 필요한 움직임을 보존하기 위해 **시작 이전 16프레임 + 종료 이후 22프레임** 추가.
- 30 FPS 기준: 앞쪽 약 0.53초, 뒤쪽 약 0.73초.
- 과거·후행 프레임을 확보한 다음 자세 분류 실행 → **반복 완료와 Good/Bad 표시 사이 추가 지연 발생**.
- 영상 시작·끝, 이전 반복과 구간 중복 등의 경계 조건을 별도로 처리.
- **코드:** `scripts/squat_rep_buffer.py`.

## 4. Good/Bad 분류 입력: 180개 Feature 상세

### 4-1. 180차원 특징 벡터를 만드는 방법

- 입력: **완료된 스쿼트 1회**의 Keypoint 시퀀스 `(T, 17, 3)`.
- 처리: 관절 좌표 → **프레임별 20개 기하학적·움직임 신호** → 반복 구간 전체에 대한 통계량 계산.
- 최종 차원:

  `기본 신호 16종 × 통계량 10종 = 160개`  
  `좌우 비대칭 신호 4종 × 통계량 5종 = 20개`  
  **`합계 = 180개`**

- **정규화:** 거리·좌표 차이는 해당 반복의 **유효한 몸통 길이 중앙값**으로 나눔. 무릎·고관절·상체 기울기 등 각도는 도(°) 단위.
- **결측값:** 신뢰도가 낮거나 검출되지 않은 관절에서 생기는 값은 `NaN`으로 유지. 분류 파이프라인에서 학습 데이터의 **특징별 중앙값**을 이용해 대체(`SimpleImputer`).
- **주의:** Feature는 관절 180개를 뜻하지 않음. **8개 주요 관절에서 계산한 20종 신호를 시간 통계로 요약한 180개 수치**임.

### 4-2. 기본 신호 16종 × 통계 10종 = 160개

아래 이름은 실제 코드에서 사용하는 Feature 이름의 **접두어(prefix)**. 각 접두어에 `mean`, `std` 등의 통계 이름이 붙어 10개씩 생성됨.

| 번호 | 신호 접두어(코드명) | 계산·의미 |
|---:|---|---|
| 1 | `left_knee_angle` | 왼쪽 **골반–무릎–발목** 각도(°). 왼쪽 무릎 굽힘 정도. |
| 2 | `right_knee_angle` | 오른쪽 골반–무릎–발목 각도(°). 오른쪽 무릎 굽힘 정도. |
| 3 | `knee_angle_mean` | 좌우 유효 무릎 각도의 평균(°). 전체적인 무릎 굽힘 정도. |
| 4 | `left_hip_angle` | 왼쪽 **어깨–골반–무릎** 각도(°). 왼쪽 고관절 굽힘 정도. |
| 5 | `right_hip_angle` | 오른쪽 어깨–골반–무릎 각도(°). 오른쪽 고관절 굽힘 정도. |
| 6 | `torso_tilt` | 어깨 중심→골반 중심 선분이 **영상의 수직축**과 이루는 각도(°). 화면에서 보이는 상체 기울기. |
| 7 | `hip_ankle_vertical` | `(발목 중심 y - 골반 중심 y) / 몸통 길이`. 골반과 발목 사이 수직 위치 차이. |
| 8 | `hip_knee_vertical_left` | `(왼쪽 무릎 y - 왼쪽 골반 y) / 몸통 길이`. 왼쪽 허벅지의 영상상 수직 성분. |
| 9 | `hip_knee_vertical_right` | 오른쪽 무릎–골반의 영상상 수직 위치 차이를 몸통 길이로 정규화. |
| 10 | `knee_ankle_horizontal_left` | `(왼쪽 무릎 x - 왼쪽 발목 x) / 몸통 길이`. 무릎–발목의 좌우 위치 차이. |
| 11 | `knee_ankle_horizontal_right` | 오른쪽 무릎–발목의 수평 위치 차이를 몸통 길이로 정규화. |
| 12 | `knee_separation` | `abs(왼쪽 무릎 x - 오른쪽 무릎 x) / 몸통 길이`. 영상상 두 무릎 사이 거리. |
| 13 | `ankle_separation` | `abs(왼쪽 발목 x - 오른쪽 발목 x) / 몸통 길이`. 영상상 두 발목 사이 거리. |
| 14 | `knee_minus_ankle_separation` | `(무릎 사이 수평 거리 - 발목 사이 수평 거리) / 몸통 길이`. 무릎과 발목 벌어짐 정도 비교. |
| 15 | `hip_relative_to_ankles_x` | `(골반 중심 x - 발목 중심 x) / 몸통 길이`. 발 위치에 대한 골반의 수평 이동. |
| 16 | `hip_y_from_start` | `(현재 골반 y - 반복 초반 15% 구간의 평균 골반 y) / 몸통 길이`. 반복 시작 위치 대비 골반 하강량. |

- **수평·수직 거리:** 이미지상 2D 투영값. 실제 공간의 3D 거리나 해부학적 각도를 직접 측정한 값이 아님.
- **16번 특징:** 초반 골반 좌표가 유효하지 않으면 계산할 수 없는 경우가 있음. 결측값은 모델의 Imputer가 처리.
- **구현 위치:** `scripts/train_squat_baseline.py`의 `extract_features()`.

#### 기본 신호마다 적용하는 시간 통계 10종

| 통계 접미어 | 의미 |
|---|---|
| `mean` | 반복 구간의 유효 프레임 평균 |
| `std` | 표준편차: 동작 중 신호의 변동 정도 |
| `min` | 최솟값 |
| `max` | 최댓값 |
| `q10` | 하위 10% 분위수 |
| `median` | 중앙값(50% 분위수) |
| `q90` | 상위 90% 분위수 |
| `range` | `max - min`: 변화 폭 |
| `first` | **반복 구간 처음 15% 프레임**의 유효값 평균 |
| `last` | **반복 구간 마지막 15% 프레임**의 유효값 평균 |

- 예시: `left_knee_angle_mean`, `left_knee_angle_min`, `left_knee_angle_last` 등.
- `first`/`last`는 **첫 프레임/마지막 프레임의 값이 아니라**, 각각 초기·말기 15% 구간의 평균.

### 4-3. 좌우 비대칭 신호 4종 × 통계 5종 = 20개

| 번호 | 신호 접두어(코드명) | 계산·의미 |
|---:|---|---|
| 1 | `asym_knee_angle` | `abs(왼쪽 무릎 각도 - 오른쪽 무릎 각도)`. 양쪽 무릎 굽힘 정도 차이(°). |
| 2 | `asym_hip_angle` | `abs(왼쪽 고관절 각도 - 오른쪽 고관절 각도)`. 양쪽 고관절 굽힘 정도 차이(°). |
| 3 | `asym_knee_motion` | `abs(왼쪽 무릎 각도 변화량 - 오른쪽 무릎 각도 변화량)`. 프레임 사이 양쪽 무릎 움직임의 비동기성(°). |
| 4 | `asym_leg_vertical` | 좌우 `(무릎 y - 골반 y) / 몸통 길이`의 **차이 절댓값**. 양쪽 다리의 영상상 수직 성분 차이. |

#### 비대칭 신호마다 적용하는 시간 통계 5종

- `mean`(평균), `std`(표준편차), `max`(최댓값), `q90`(90% 분위수), `range`(최대–최소 변화 폭).
- 예시: `asym_knee_angle_mean`, `asym_knee_motion_q90`, `asym_leg_vertical_range`.
- **구현 위치:** `scripts/train_squat_extended.py`의 `extract_feature_groups()`에서 `asymmetry` 그룹 생성.

### 4-4. 실험한 다른 특징과 최종 모델의 차이

- `train_squat_extended.py`에는 `alignment`, `phase`, `trajectory`, `quality` 등 **다른 실험용 특징 그룹도 존재**.
  - `alignment`: 무릎–발목의 정면 투영 정렬 관계 등.
  - `phase`: 하강·최저점·상승 구간별 특징.
  - `trajectory`: 특정 동작 시점의 특징.
  - `quality`: 관절 검출 품질 관련 지표.
- **현재 배포용 모델에 실제 사용하는 그룹은 `base`(160개) + `asymmetry`(20개)뿐**.
- 새로운 특징을 추가하면 기존 모델과 입력 차원·순서가 달라질 수 있으므로 **모델 재학습과 평가 필요**.

## 5. Good/Bad 분류기와 학습 방식

### 5-1. 최종 모델 구성

| 항목 | 내용 |
|---|---|
| 파일 | `models/squat_plus_asymmetry180_crop10_all9.joblib` |
| 모델 | Logistic Regression(이진 분류) |
| 입력 | `PlusAsymmetry180`: 180차원 실수 벡터 |
| 전처리 | `SimpleImputer(strategy='median', keep_empty_features=True)` → `StandardScaler()` |
| 분류기 설정 | `LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000, random_state=42)` |
| 정답 라벨 | `0 = Bad`, `1 = Good` |
| 예측 시점 | **스쿼트 1회 완료 후** |

- **Feature 이름과 순서:** 모델 파일의 `feature_names` 목록을 반드시 사용. 180개 숫자의 **순서가 바뀌면 올바르게 예측할 수 없음**.
- **결측값 대체·표준화:** 학습된 `pipeline` 안에 포함됨. 보드 이식 시 임의로 다른 방식으로 재구현하지 않도록 주의.
- **직접 자세 오류 진단은 불가능:** 현재 모델의 출력은 `Good` 또는 `Bad` 한 가지이며, 무릎 모임·상체 숙임 등 원인 종류를 구분하지 못함.

### 5-2. 시간 구간 절단 증강(Crop10)

- **목적:** 자동 카운터가 검출한 시작·종료 구간이 정답 구간과 조금 다르더라도 자세 분류 성능을 유지하도록 학습.
- **한 반복에서 4종 시퀀스 생성:**
  1. 원본 전체 구간.
  2. 시작 10% 제거.
  3. 종료 10% 제거.
  4. 시작·종료 각각 10% 제거.
- **학습 데이터 규모:** 실제 스쿼트 **195회 × 4종 변형 = 780개 학습 특징 행**.
- **새로운 독립 반복 780회가 생겼다는 의미는 아님:** 동일한 195회를 변형한 데이터.
- **모델 메타데이터 구분:**
  - `feature_variant='PlusAsymmetry180'`: **특징 구성**(180개).
  - `training_variant='PlusAsymmetry180_crop10_all9'`: **학습 방식**(Crop10, 전체 9명).
- **코드:** `scripts/train_squat_crop10_all9.py`, `scripts/compare_squat_lopo_augmentation.py`.

### 5-3. LOPO 평가와 실제 배포 모델의 차이

- **LOPO(Leave-One-Person-Out):** 9명 중 **8명으로 학습**, 남은 **1명으로 테스트**하는 과정을 9번 반복.
- **86.2%는 LOPO 평가 결과**이며, **전체 9명으로 학습한 최종 `.joblib`을 같은 9명에게 다시 적용한 정확도가 아님**.
- 최종 `.joblib`은 **실제 프로토타입 실행을 위해 9명 전원 데이터로 다시 학습한 모델**.
- 모델·증강 방법 선택에 같은 데이터셋의 반복 실험을 사용했으므로, 완전히 독립적인 신규 사용자 검증은 추후 별도 수행해야 함.

## 6. 주요 실험 결과와 개선 과정

### 6-1. 카운팅 및 분류 성능

**대상:** REHAB24-6 Camera17, 30 FPS, 피험자 9명, 총 195회.

| 평가 항목 | 결과 | 해석 |
|---|---:|---|
| 스쿼트 반복 검출 | **195/195회** | 평가 영상의 정답 반복과 자동 검출 반복이 모두 대응 |
| 평균 반복 구간 IoU | **0.860** | 자동 검출 구간과 정답 구간의 겹침 정도. **14% 짧다는 의미는 아님** |
| Good/Bad 분류 정확도(LOPO) | **168/195 = 86.2%** | 195회 중 168회 정답 |
| Bad 재현율 | **54/61 = 88.5%** | 실제 Bad 61회 중 54회 검출 |
| Bad 정밀도 | **54/74 = 73.0%** | Bad라고 판단한 74회 중 54회 정답 |
| Bad F1 | **약 80.0%** | Bad 정밀도·재현율의 조화평균 |

| 실제 라벨 \ 예측 | Good | Bad |
|---|---:|---:|
| 실제 Good(134회) | **114** | 20(오경고) |
| 실제 Bad(61회) | 7(놓침) | **54** |

- **주의:** 카운팅 195/195와 분류 정확도 86.2%는 **서로 다른 지표**.
- **주의:** 기존 Camera17의 개발·모델 선택에 반복 사용된 평가 결과. 독립 외부 데이터의 성능을 의미하지 않음.

### 6-2. 문제 해결 기록(중간 발표 내용)

| 발견한 문제 | 적용한 개선 | 확인한 결과 |
|---|---|---|
| 배경 인물로 운동 수행자 추적이 바뀜 | 수동 Anchor + 프레임 간 박스 위치·IoU 비교 | 수행자 관절 추출 안정화 |
| 자동 반복 구간의 일부 누락 | 앞 16프레임 / 뒤 22프레임 보완 | PM_043의 Bad 탐지 **4/10 → 10/10** |
| 얕은 스쿼트 검출 누락 | 최저점 골반 깊이 `0.45 → 0.20` | PM_126의 카운팅 **14/20 → 20/20** |
| 피험자별 Good/Bad 분류 성능 편차 | 시간 절단 증강 Crop10 | 정확도 **83.6% → 86.2%** |

- **비교 주의:** 위 개선 수치는 실험마다 평가 조건이 다르며, 일부 개별 영상 결과는 전체 피험자를 사용한 모델을 통한 원인 분석 결과. 전체 LOPO 성능과 직접 동일한 조건의 비교로 해석하지 말 것.

### 6-3. 속도·메모리(측정 환경 구분)

- **개발 PC(RTX 4070 Ti):** 저장된 영상 1,200프레임 처리에서 **83.2 FPS**. PC 기반 측정치.
- **R2130-12 + Hailo-8:** Pose 샘플 추론 실행 확인, 샘플 구동 시 시스템 RAM 사용량 **약 289.2 MiB 증가**(중간 발표 측정).
- **아직 미측정:** R2130-12에서 실제 카메라 입력을 포함한 전체 AI 파이프라인 FPS, 반복 완료→분류→앱 표시 지연, 장시간 메모리·안정성.
- **지연 특성:** 종료 후 22프레임(30 FPS에서 약 0.73초) 확보가 필요하므로 FPS와 별개로 Good/Bad 출력이 늦어질 수 있음.

## 7. AI ↔ 서버 ↔ 앱 이벤트 형식과 통합 방법

### 7-1. 현재 AI 코드의 실제 출력

- **생성 위치:** `results/stream_test/<영상ID>_buffered_events.jsonl` (PC 원본 MP4를 분석한 결과).
- **생성 시점:** FSM이 스쿼트 1회를 검출하고, 후행 버퍼를 수집해 Good/Bad 분류를 완료한 시점.
- **현재 제공하는 이벤트:** `type="rep"`, `exercise="squat"` 중심. 실제 서버로 POST하는 부분은 **아직 연결되지 않음**.

| 필드 | 의미 | 현재 AI 구현 |
|---|---|---|
| `type` | 이벤트 유형 | `rep` |
| `exercise` | 운동 종류 | `squat` |
| `count` | 누적 완료 반복 횟수 | 구현 |
| `posture` | 반복의 Good/Bad | `good` / `bad`, 추론 실패 시 `unknown` |
| `issues` | 세부 자세 오류 코드 | **항상 `[]`**. 개별 오류 진단 미구현 |
| `score` | 자세 점수 | **`null`**. 점수 모델 미구현 |
| `start_frame`, `end_frame` | 분류에 사용한 보완 구간 | 구현 |
| `detected_start_frame`, `detected_end_frame` | FSM이 탐지한 원래 구간 | 구현 |
| `classification_frame_count` | 분류 입력 프레임 수 | 구현 |
| `post_context_complete` | 종료 뒤 버퍼 수집 완료 여부 | 구현 |
| `probability_correct_uncalibrated` | Good 예측 확률 | 구현. **신뢰도 보정된 자세 점수가 아님** |

**AI 반복 이벤트 예시(실제 측정 로그가 아닌 구조 예시)**

```json
{
  "type": "rep",
  "exercise": "squat",
  "count": 1,
  "posture": "good",
  "issues": [],
  "score": null
}
```

- `unknown` 및 `classification_error` 등 실패 이벤트를 서버·앱이 어떻게 처리할지 별도 합의 필요.
- 서버 입력 스키마가 위 필드 중 어떤 것을 허용·필수로 처리하는지는 **통합 시 서버 담당자와 최종 확인 필요**.

### 7-2. 서버 구현 현황 — 손준표 중간 보고서 기준

- **실행 환경:** Raspberry Pi의 FastAPI + ngrok 고정 도메인.
  - 앱은 WebSocket `/ws` 경로로 실시간 접속. 같은 핫스팟 외의 외부 네트워크(LTE 등)에서도 연결 성공.
  - Raspberry Pi IP가 바뀌어도 ngrok 고정 도메인을 사용하므로 앱 측 IP 재입력 불필요.
  - **서버와 ngrok을 `systemd` 서비스로 등록**: 부팅 시 자동 실행, 오류 시 자동 재시작 및 재부팅 후 앱 연결 확인.
  - **ngrok 터널과 서버 프로세스가 모두 실행 중**이어야 앱에서 접속 가능.
- **프로토콜 및 검증:**
  - 이벤트 종류: `session_start`, `rep`, `posture`, `session_end` 설계.
  - 공통 필드: 운동 종류, 반복 횟수, 정오 판정, 피드백 코드 등의 JSON 구조 정의.
  - 서버가 요청 형식을 검사하여 잘못된 데이터는 HTTP `422`로 거절하도록 구현.
  - `random_sender`: 세션 시작 → 반복 → 자세 경고 → 세션 종료 순서의 **가상 이벤트**를 생성하는 테스트 도구.
  - 브라우저 확인 페이지: 서버 수신 기록의 실시간 표 표시·디버깅.
- **남은 작업:** AI 출력과 서버 입력 형식 최종 일치, 실제 AI POST 연동, R2130-12에서 AI와 서버를 통합 실행하는 구성 검증, 동일 네트워크 직접 접속을 예비 경로로 시험.

### 7-3. 앱 구현 현황 — 임태빈 중간 보고서 기준

| 앱 파일(앱 프로젝트 기준) | 역할 및 확인된 동작 |
|---|---|
| `src/config/serverConfig.js` | ngrok 고정 도메인, `WS_URL`(`/ws`), `RECORDS_URL`(`/records`), 관련 헤더를 한곳에서 관리 |
| `src/context/WebSocketContext.js` | 앱 실행 시 WSS 자동 연결, 3초 후 재연결, 백그라운드 복귀 시 재접속, `connectionId`로 오래된 소켓 이벤트 방지 |
| `src/app/index.js` | 스쿼트·푸시업 선택, 연결 상태 확인, 재연결 버튼; 미연결 상태에서 운동 시작 제한 |
| `src/app/workout.js` | 실시간 횟수·자세 표시, 운동 종료 시 **휴대폰 SQLite에만 저장** |
| `src/db/database.js` | `expo-sqlite` 기반 로컬 운동 기록 저장 |
| `src/app/_layout.js` | 운동 선택 및 연결 여부에 따른 화면 접근 제한 |

- **WebSocket 수신 형식:** 현재 앱은 **낱개 이벤트**와 **기록 목록 전체 재전송**을 모두 처리하도록 구성.

  **낱개 이벤트 예시(앱 파싱 지원 형식, 실제 AI 출력 아님)**

  ```json
  {"count": 12, "posture": "good", "issues": [], "score": 90}
  ```

  **서버에서 확인된 목록 메시지 형식(아래 ID·값은 설명용 예시)**

  ```json
  {
    "total": 2,
    "records": [
      {"id": 101, "type": "rep", "exercise": "squat", "count": 2, "posture": "good", "issues": [], "score": null},
      {"id": 102, "type": "posture", "exercise": "squat", "count": 2, "posture": "bad", "issues": ["back_lean"], "score": null}
    ]
  }
  ```

  - 목록 재수신 시 기존에 처리한 `id`는 **중복 반영하지 않고 건너뜀**. 앱에는 새 기록만 순서대로 적용.
  - `type="rep"`: **카운트·Good/Bad 집계 반영**.
  - `type="posture"`: **피드백 표시만 갱신**. 카운트·정확도 집계에는 반영하지 않음.
  - `session_start` / `session_end`: 현재 앱은 무시하며, 사용자의 운동 선택·종료 버튼으로 세션을 관리.
  - **운동 기록 저장:** 앱에서 운동을 종료할 때 SQLite 로컬 저장. 서버 `/records`로 운동 기록을 다시 전송하지 않음. 서버의 `GET /records?after={id}`는 **현재 앱에서 미사용**.
  - **운동 종류 필터:** 현재 앱이 다른 운동(`exercise`) 이벤트를 걸러내지 않는 상태 → 통합 시 보완 필요.

- **앱에 정의된 오류 코드 → 한국어 표시:**

| `issues` 코드 | 앱 표시 문구 | 현재 AI 생성 여부 |
|---|---|---|
| `knee_over_toe` | 무릎이 발끝보다 앞으로 나갔어요 | 미구현 |
| `back_lean` | 허리가 너무 숙여졌어요 | 미구현 |
| `shallow_depth` | 너무 얕게 앉았어요 | 미구현 |
| `hip_sagging` | 엉덩이가 처졌어요 | 미구현 |

- 위 코드들은 **앱의 피드백 문구 매핑과 서버 테스트 이벤트 규격**이며, 현재 AI 모델이 실제 운동에서 판정하는 항목이 아님.
- 앱 쪽에서 확인이 필요한 사항: Expo Go에서 ngrok 우회 헤더 적용 여부, `/ws`의 최종 이벤트/목록 전송 정책, 운동 종류 필터링, 추가 오류 코드.

### 7-4. 통합 시 확인할 사항

| 확인 항목 | 담당 간 합의할 내용 |
|---|---|
| **AI → 서버 연결** | AI의 `rep` JSON을 서버 FastAPI 입력 경로에 실제 POST하는 어댑터 추가; 엔드포인트·필수 필드·HTTP 응답 규격 확인 |
| **서버 → 앱 전송** | `/ws`에서 단일 이벤트 또는 `records` 배열 중 어떤 방식으로 보낼지 확정; 두 형식 모두 앱에서 처리 가능하도록 작성됨 |
| **중복 방지** | 서버가 전송하는 각 기록의 고유 `id`, 전송 순서, 재연결 후 목록 재송신 정책 유지 |
| **카운트·세션** | `rep`만 카운트 반영; 앱은 현재 세션 시작·종료를 자체 관리하므로 역할 충돌 여부 확인 |
| **오류·점수** | AI의 `issues=[]`, `score=null`, `unknown` 처리를 허용할지 확인; 실제 오류 진단·점수 기능은 향후 개발 |
| **종목 구분** | 앱이 선택한 운동과 수신한 `exercise`가 다르면 무시하는 로직 검토 |
| **네트워크·운영** | ngrok 주소는 앱 `serverConfig.js`에서 관리, 서버·터널 실행 상태 및 오프라인 안내 확인 |
| **실제 하드웨어** | Hailo Pose 출력의 관절 순서/좌표계·신뢰도 변환, R2130-12 구동 시 FPS·지연·메모리 실측 |

- **참고 노션 페이지:** [손준표 — 서버 중간 보고서](https://app.notion.com/p/3f8e128967e280a3925ad462867a6777), [임태빈 — 앱 중간 보고서](https://app.notion.com/p/3f8e128967e28028b146edffd4ac5adc).
- 위 사항은 **2026-10-13경 작성된 팀원 중간 보고서 기준**. 이후 서버·앱 수정 사항이 있다면 해당 저장소와 팀원 확인을 우선할 것.

## 8. 개발 환경, 실행 방법, 주요 파일

### 8-1. 개발 환경

- **PC/WSL:** `capstone_pose` Conda 환경, Python 3, Ultralytics, PyTorch, OpenCV, NumPy, scikit-learn, joblib 등.
- **R2130-12:** Debian 12(Bookworm) 기반 ARM64, HailoRT 4.20.0 환경에서 Hailo 샘플 구동 확인.
- **환경 목록:** `requirements-wsl.txt`(PC 환경 기록용; ARM64에 그대로 설치하는 용도가 아님).
- **추가로 필요한 파일:** REHAB24-6 원본 영상 및 정답 파일, `yolov8m-pose.pt`(일반 WSL 추론용 YOLO 가중치). 원본 MP4·모델 가중치 중 일부는 GitHub에 없음.

### 8-2. PC에서 최신 스쿼트 분석 실행

```bash
cd ~/capstone2/pose_training
conda activate capstone_pose

# 실행 옵션 확인
bash scripts/run_squat_latest.sh --help

# PM_029 원본 영상 전체를 순차 분석
bash scripts/run_squat_latest.sh --video PM_029 --max-frames 0

# 테스트 실행
python -m pytest -q tests
```

- **입력 파일 기본 경로:** `data/raw/REHAB24-6/PM_029-Camera17-30fps.mp4`.
- **출력 기본 경로:** `results/stream_test/PM_029_buffered_events.jsonl`.
- **Anchor 설정:** `configs/rehab24_squat_camera17_subjects.json`.
- **최신 실행 스크립트:** `scripts/run_squat_latest.sh`는 `--model ...crop10_all9.joblib`, `--bottom-depth 0.20`, `--pre-frames 16`, `--post-frames 22`를 지정하여 기존 실험 실행기에 전달.
- **실행 옵션으로 값을 변경하면 평가 당시 조건과 달라질 수 있음.** 변경된 옵션은 실험 기록에 함께 기재.
- **테스트 현황:** 2026-10-14 기준 `13 passed`. 단위 테스트가 영상 전체 성능이나 실제 보드 동작을 보장하지는 않음.

### 8-3. 저장소의 핵심 파일

| 경로 | 역할 |
|---|---|
| `scripts/run_squat_latest.sh` | 최신 모델·임계값·버퍼 설정 실행 래퍼 |
| `scripts/run_squat_video_stream_tunable.py` | 원본 MP4를 프레임별로 분석해 반복 이벤트 출력 |
| `scripts/stream_squat_core.py` | 골반 이동·무릎 각도 기반 상태 전이 카운터 |
| `scripts/squat_rep_buffer.py` | 반복 구간 앞뒤 프레임 보완 |
| `scripts/extract_all_squats_tracked.py` | 학습·평가용 관절 시퀀스 추출 및 대상 추적 |
| `scripts/train_squat_baseline.py` | 기본 16개 신호 / 160개 특징 및 초기 분류기 실험 |
| `scripts/train_squat_extended.py` | 비대칭 신호 포함 확장 특징 계산 및 실험 |
| `scripts/run_squat_video_stream.py` | 180개 특징 구성 및 Logistic Regression 추론 함수 `classify()` 포함 |
| `scripts/train_squat_crop10_all9.py` | Crop10 최종 모델 학습 |
| `scripts/compare_squat_lopo_augmentation.py` | LOPO 및 Crop 증강 실험 |
| `scripts/validate_squat_stream.py` | 추론 완료 후 정답 구간과 카운팅·분류 결과 비교 |
| `scripts/render_squat_presentation_demo.py` | 발표용 영상 렌더링(저장된 예측 결과 시각화) |
| `models/squat_plus_asymmetry180_crop10_all9.joblib` | 9명 전체로 재학습한 최종 이진 분류기 |
| `configs/rehab24_squat_camera17_subjects.json` | 영상별 수동 지정 대상 Anchor |
| `docs/STREAMING.md` | 초기 프로토타입 및 스트리밍 구현 관련 기록 |
| `tests/` | 로직별 단위 테스트 |

- **시연 MP4:** 저장된 실제 추론 결과(JSONL)를 원본 영상과 동기화해 표시한 영상. **시연 MP4 재생 자체가 매번 Good/Bad 추론을 수행하는 것은 아님**.

## 9. 향후 개발 우선순위와 협업 체크리스트

### 9-1. AI/보드 통합

- [ ] Hailo-8 Pose 결과를 현재 코드가 기대하는 **COCO17 `(17, 3)`의 `(x, y, confidence)` 형식**으로 변환.
- [ ] Hailo 출력의 좌표계·정규화 방식·관절 순서·신뢰도 기준 확인. **PC Ultralytics 출력과 다르면 반드시 변환**.
- [ ] 여러 사람 중 분석 대상 자동 선택 및 추적, 가림·화면 이탈 시 동작 확인.
- [ ] 기존 SquatCounter와 Crop10 분류기 연결. `feature_names` 순서 및 전처리 유지.
- [ ] USB 카메라 입력에서 전체 FPS·메모리·종료 후 판정 지연 실측.

### 9-2. 서버·앱 연동

- [ ] 현재 AI의 JSONL 출력을 FastAPI 입력 형식으로 변환하고, 실제 HTTP POST 송신 기능 구현.
- [ ] FastAPI의 JSON 검증(`422` 오류 포함) 통과 여부를 실제 AI 이벤트로 확인.
- [ ] 서버 `/ws`의 **낱개 이벤트 / `records` 배열** 전달 방식에서 `rep` 카운트 중복 방지 및 기록 `id` 처리 검증.
- [ ] `posture` 이벤트는 피드백만, `rep` 이벤트는 횟수·정오 집계만 변경하도록 테스트.
- [ ] 앱 SQLite 로컬 저장 및 세션 시작/종료 버튼의 책임 범위 확인. 서버로 기록 재전송하는 기능은 현재 미사용.
- [ ] 현재 AI의 `issues=[]`, `score=null`, `unknown` 처리와 앱의 `ISSUE_LABELS` 확장 규칙 합의.
- [ ] 앱에서 **선택 운동과 다른 `exercise` 이벤트 무시** 기능 검토.
- [ ] ngrok 접속·자동 재연결·서버 재시작 및 동일 핫스팟 예비 경로 검증.

### 9-3. 성능 개선·추가 검증

- [ ] REHAB24-6 **Camera18**: 학습된 모델·임계값 고정 후 카메라 시점 변화 평가.
- [ ] 처음 보는 사용자와 실제 카메라 영상에서 검증.
- [ ] 스쿼트 외 움직임에서의 **잘못된 카운팅(False Reps)** 측정.
- [ ] Good 오경고·Bad 놓침을 피험자·카메라 각도별로 분석.
- [ ] 실제 오류 종류별 라벨이 확보되면 `issues` 진단·교정 문구 기능 확장.
- [ ] 푸시업 등 운동 종류 확장 시 별도 카운터·특징·학습 데이터 설계.

## 10. 데이터·모델·Git 관리 주의사항

- **Git 업로드 제외:** `data/`, `results/`, `logs/`, 원본 MP4, `*.npz`, `*.csv`, 대용량 Pose 가중치 등(`.gitignore` 확인).
- **데이터 재배포 금지:** REHAB24-6 제공 조건을 확인하고 원본 영상·정답 파일을 GitHub에 올리지 말 것.
- **Joblib 보안:** 신뢰할 수 있는 자체 학습 파일만 불러올 것(역직렬화 시 코드 실행 위험).
- **성능 기록:** **데이터셋/카메라/피험자 분리/모델/임계값/버퍼/측정 장치**를 함께 명시할 것.
- **혼동 금지:** `195/195 카운팅` ≠ `86.2% Good/Bad 분류`; `PC 83.2 FPS` ≠ `R2130-12 실측 FPS`; `Crop10 학습 780행` ≠ `실제 운동 반복 780회`.
- **실험 기록과 현재 구현 분리:** `docs/STREAMING.md`의 과거 프로토타입 설명과 이 README의 최신 실행 설정을 구분할 것.


## 11. Pose 입력과 스쿼트 분석 파이프라인 분리 (2026-10-28, 리팩터링)

- **변경 목적:** YOLOv8m-Pose 추출부와 SquatCounter → RepBoundaryBuffer → 180-feature LR 분류 → 'rep' JSON 출력을 분리해, 향후 Hailo-8의 COCO17 관절을 재사용 가능한 분석부에 연결.
- **공통 모듈:** 'scripts/squat_stream_pipeline.py'의 'SquatStreamPipeline'을 사용. 프레임마다 원본 영상 **픽셀 좌표계**의 'float32 (17,3)' '[x,y,confidence]' 배열을 'push(kp, frame_id=...)'에 넣으면, 완성된 'rep' 이벤트 딕셔너리 목록을 반환함. 마지막에는 'flush()' 호출.
- **분류기 분리:** 'scripts/squat_classification.py'의 'classify(seq, artifact)'가 기존 180개 특징/학습된 sklearn Pipeline을 그대로 사용. 새로운 학습은 하지 않음.
- **기존 실행:** 'scripts/run_squat_video_stream_tunable.py'는 여전히 YOLO/수동 Anchor로 keypoint를 얻지만 분석과 JSON 생성은 공통 모듈에 위임. 'scripts/run_squat_latest.sh'의 'bottom_depth=0.20', 'pre_frames=16', 'post_frames=22' 설정은 유지.
- **미완료:** Hailo 출력 형식·좌표 변환·수행자 선택·USB 카메라·AI→FastAPI POST·실기기 성능은 별도 통합 단계. 이 변경만으로 Hailo 추론이나 서버/앱 연결이 구현되는 것은 아님.

예시 (이미 출력 형식을 검증한 Hailo 키포인트 배열이 있다고 가정):

```python
import joblib
from squat_stream_pipeline import SquatStreamPipeline, missing_pose

artifact = joblib.load("models/squat_plus_asymmetry180_crop10_all9.joblib")  # 자체 생성 파일만
pipeline = SquatStreamPipeline(artifact=artifact, fps=30, bottom_depth=0.20,
                               pre_frames=16, post_frames=22)

# 프레임마다 한 번씩 수행. 검출 실패 시에도 프레임을 건너뛰지 않음.
for frame_id, kp in source_of_selected_coco17_frames:
    for event in pipeline.push(kp if kp is not None else missing_pose(), frame_id=frame_id):
        print(event)  # 추후 서버 POST로 교체
for event in pipeline.flush():
    print(event)
```

테스트: 'python -m pytest -q tests/test_squat_stream_pipeline.py'; 기존 영상의 JSONL 이벤트를 리팩터링 전과 후에 비교해 최종 회귀 검증할 것.


## 12. Hailo GStreamer Pose → COCO17 입력 어댑터 (2026-10-28)

- **코드:** \`scripts/hailo_pose_adapter.py\`
- **테스트:** \`tests/test_hailo_pose_adapter.py\` (HailoRT 없이 가짜 Detection/BBox/Point로 검증)
- **기준:** [Hailo 공식 Pose callback](https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/basic_pipelines/pose_estimation.py)의 Bounding Box 상대 좌표 변환 방식.
- **입력:** \`hailo.HAILO_DETECTION\` 객체, 실제 **callback pad caps에서 얻은 영상 width/height**, \`hailo.HAILO_LANDMARKS\` 타입 상수
- **출력:** \`float32 (17, 3)\` 배열, COCO17 인덱스 순서 \`[x_pixel, y_pixel, confidence]\`; 신뢰도 0.30 미만 관절 좌표는 \`NaN\`. 모델 학습이나 Hailo 라이브러리 import 불필요.
- **1인 MVP 정책:** 화면에 'person' 검출이 정확히 한 명일 때만 변환. 0명·2명 이상이거나 landmarks가 없으면 \`None\` 반환; 호출자가 \`missing_pose()\`로 바꿔 매 프레임을 빠짐없이 파이프라인에 넣어야 함.

실제 Hailo \`app_callback\` 내부에서 연결할 코드 예시 (**아직 R2130-12에서 실행 검증되지 않음**):

```python
from hailo_pose_adapter import single_person_coco17
from squat_stream_pipeline import missing_pose

# callback 내부. 이미 생성한 pipeline 객체를 재사용할 것.
fmt, width, height = get_caps_from_pad(pad)
if width is not None and height is not None:
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    kp = single_person_coco17(
        detections, width, height, landmarks_type=hailo.HAILO_LANDMARKS
    )
    if kp is None:
        kp = missing_pose()
    for event in pipeline.push(kp, frame_id=frame_id):
        print(event)
```

**제한과 현장 확인:** 
- 예시는 콜백 함수 전체가 아님. \`pipeline\`은 모델을 로드해 콜백 전에 초기화하고, \`frame_id\`는 0부터 프레임마다 순서대로 증가해야 함. 종료 시 \`pipeline.flush()\` 호출.
- 영상 좌표는 예제 **callback 버퍼의 크기** 기준임. 카메라 원본 영상과 callback 버퍼 사이에 crop/letterbox/resize/rotate가 있다면 그 좌표 매핑부터 실제 보드에서 확인. 무조건 카메라 원본 해상도에 곱하면 안 됨.
- 단일 인물/관절 없는 프레임 처리만 구현함. 추후 추적 ID 및 다중 인물 추적, 분류 성능 변화, 처리 FPS/지연시간은 보드에서 별도 검증.
- 변환 함수와 단위 테스트를 추가한 것일 뿐 **GStreamer 실행 코드에 자동으로 연결되지는 않음**.

로컬 검증: \`python -m pytest -q tests/test_hailo_pose_adapter.py tests/test_squat_stream_pipeline.py\`
