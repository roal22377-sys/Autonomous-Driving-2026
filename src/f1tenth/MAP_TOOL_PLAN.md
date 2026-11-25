# 실제 코스 지도 따기 → 시뮬레이터 트랙 변환 도구 계획 (11/25 정리)

실차가 달리는 실제 코스(지금은 연습 코스, 나중에는 대회 코스)를 정확한 지도로 만들고, 그 지도를 시뮬레이터 트랙 파일로
바꾸기 위한 정보와 계획입니다. **아직 도구는 만들지 않았습니다.** 이 문서는 무엇이 필요하고 어떤 순서로 만들지를 정리한 것입니다.
모든 명령은 저장소 루트(`~/Autonomous-Driving-2026`) 기준입니다.

---

## 1. 왜 필요한가

| 쓰임 | 설명 |
|---|---|
| 검증용 트랙 | 학습에 안 쓰는 "실제 모양" 트랙 하나를 평가에 넣어, 모델이 실제 코스 모양에서도 되는지 확인 (`compare_models.py`에 별도 묶음) |
| 실차 주행 분석 | 지도 위에 실차 궤적·사고 지점을 정확히 그려 원인 찾기 |
| 대회 코스 대비 | 대회 당일 코스를 빨리 지도화 → 시뮬레이터에서 expert/모델 확인, 필요하면 그 코스로 짧게 미세조정 |

연습 코스는 대회 코스와 다르므로, 연습 코스 트랙은 **학습용이 아니라 검증용**입니다(생성 트랙 920개로 학습하는 방식은 그대로).

---

## 2. 지금 있는 것과 한계

### `tools/log_map.py` (11/24에 쓴 지도)
- 방식: 주행 로그(`logs/<run>_extracted/scan.npz`)의 LiDAR 스캔을 **바로 앞까지 만든 지도에 ICP로 이어 붙임**
  (odom 속도가 11/24 오전에 1.35배 틀려 있어서 odom 대신 스캔 매칭을 씀).
- 결과 예: `logs/run_1124_1521_extracted/analysis/map_full.png`
- **트랙으로 못 쓰는 이유**
  - 오차가 계속 쌓이고, 한 바퀴 돌아왔을 때 위치를 다시 맞추는 **루프 클로저가 없음**
  - 그래서 **벽이 2~3겹(약 0.3~0.5 m씩 어긋남)**, 바퀴마다 궤적이 어긋남, 일부 번짐
  - 차가 제자리에서 돌거나 막히면 매칭 실패
- 용도: "어디서 사고가 났는지" 보는 분석용으로는 충분, 정밀 트랙으로는 부족.

### 필요한 정밀도
- 시뮬레이터 트랙의 통로 반폭은 약 1 m, 차 반폭 0.15 m, 레이스 라인은 벽에서 0.55 m(`RACELINE_MARGIN`).
- 벽 위치가 **10~20 cm만 틀려도** 레이스 라인·충돌 판정이 달라지므로, 벽 오차 **±5 cm 안팎**이 목표.

---

## 3. 시뮬레이터 트랙 형식 (변환 결과가 맞춰야 할 것)

생성 트랙(`src/f1tenth/installation/maps_gen/*.npz`, `utils/track_gen.py`)과 같은 `.npz`이면 학습·평가 코드가 그대로 읽습니다
(`utils/track_gen.py` `to_track`).

| 필드 | 형태 | 내용 |
|---|---|---|
| `occupancy` | (H, W) uint8 | 점유 격자 (>128 = 빈 공간). 충돌 판정(`env.footprint_in_wall`)과 LiDAR 시뮬레이션에 씀 |
| `resolution` | float | [m/칸] 생성 트랙은 0.08 (`RESOLUTION`) |
| `origin` | (3,) | 격자 (0,0) 칸의 월드 좌표 [x, y, yaw] |
| `centerline` | (N, 2) | 통로 중심선, **0.1 m 간격**(`DENSE_DS`), 닫힌 고리, 주행 방향 순서 |
| `w_left`, `w_right` | (N,) | 중심선 각 점에서 왼쪽/오른쪽 벽까지 거리 [m] (막다른 공간·오목한 곳 제외한 "매끈한 통로" 폭; `--bay_cost`가 이 값을 씀) |
| `raceline` | (M, 2) | 레이스 라인 (`utils/raceline.py` `min_curvature_line` → `refine_min_time`, 벽에서 0.55 m) |
| `length` | float | 중심선 길이 [m] |
| `runoffs` | (K, 6) | 코너 입구 막다른 공간 사각형 (실제 코스는 빈 배열로 둬도 됨) |
| `name`, `map`, `reverse`, `original`, `gen_sig` | | 이름·지도·방향 표시 (실제 코스는 `original=True`, `map='Practice1124'` 등) |

양방향을 모두 쓰려면 정방향·역방향(`_rev`) 두 파일을 만듭니다(중심선 순서와 좌우 폭만 뒤집음).

---

## 4. 계획

### 단계 A. 차에서 SLAM으로 지도 만들기 (루프 클로저 있음)
차는 **ROS 2 Foxy**(`source /opt/ros/foxy/setup.bash`, `tools/drive_session.sh`)이고, 지금 저장소에는 SLAM 설정이 없습니다.

1. 설치 (차에서, 한 번):
   ```bash
   sudo apt install ros-foxy-slam-toolbox ros-foxy-nav2-map-server
   ```
2. 필요한 TF는 이미 있음:
   - `odom → base_link`: `vesc_to_odom_node` (`config/vesc.yaml`, `publish_tf: true`)
     - 11/24 보정 뒤 `speed_to_erpm_gain: 3100`으로 odom 속도가 맞음. 보정 전 로그는 SLAM에 쓰기 어려움
   - `base_link → laser`: `bringup_launch.py`의 static transform (0.13, 0, 0.153)
   - 스캔: `/scan`, frame `laser` (`config/sensors.yaml`)
3. 실행: bringup + 키보드 조종 + slam_toolbox(online async, mapping 모드).
   설정 파일 초안(`config/slam.yaml`로 추가 예정, Foxy 버전 인자 이름은 차에서 확인):
   ```yaml
   slam_toolbox:
     ros__parameters:
       mode: mapping
       odom_frame: odom
       base_frame: base_link
       map_frame: map
       scan_topic: /scan
       resolution: 0.05          # 저장 후 0.08로 다시 샘플링 가능
       max_laser_range: 10.0     # Hokuyo UST-10LX 최대
       minimum_travel_distance: 0.1
       minimum_travel_heading: 0.1
       do_loop_closing: true
   ```
4. 주행: 키보드로 **1.0~1.5 m/s 이하**, 코스 1~2바퀴. 출발 지점으로 다시 돌아와 루프가 닫히게 합니다.
   갈림길이나 막다른 곳이 있으면 그 안까지 한 번 들어갔다 나옵니다.
5. 저장:
   ```bash
   ros2 run nav2_map_server map_saver_cli -f ~/maps/practice_1124
   ```
   결과는 `practice_1124.pgm`과 `.yaml`입니다. 같은 세션의 bag도 꼭 남깁니다(`/scan /odom /tf /tf_static`). 그러면 나중에 오프라인으로 다시 지도를 만들 수 있습니다.
   - 지금 `drive_session.sh`의 bag에는 `/tf`, `/tf_static`이 없으므로, 지도 세션용으로 추가가 필요합니다.
6. 확인: 줄자로 통로 폭 3~4곳과 직선 길이 1곳을 재서 지도와 비교합니다. 오차 5 cm 안팎이 목표입니다.

**미리 확인할 위험**
- LiDAR 기울기(11/24 오전 2~3° 아래로 기울어 바닥 반사 → 수평 맞춤 필요, `CAR_TESTS_1124.md` 시험 0)
- 유리·반사 벽, 통로 사이 틈(연습 코스는 틈을 막고 시험함)
- 10 m 밖 벽은 안 보임 → 넓은 공간에서는 천천히 가까이 지나가기

### 단계 B. 변환 도구 `tools/map_to_track.py` (만들 예정)
SLAM 지도(pgm/yaml) → 시뮬레이터 트랙 `.npz`:
1. **불러오기:** pgm + yaml(resolution, origin, 임계값) → 점유 격자. 0.08 m/칸으로 다시 샘플링합니다.
2. **정리:** 작은 구멍과 점 잡음을 없애고(형태학적 열기·닫기), 지도 밖이나 알 수 없는 칸은 벽으로 둡니다.
3. **주행 영역:** 출발 위치(또는 사용자가 찍은 한 점)에서 이어진 빈 공간만 남깁니다.
4. **중심선:** 빈 공간의 거리 변환 → 골격선(skeleton). 가장 큰 고리 하나만 남기고 가지(막다른 곳)는 잘라냅니다. 그다음 순서대로 정렬하고, 부드럽게 다듬고(`utils/raceline.py` `resample_closed`), 0.1 m 간격으로 만듭니다.
5. **폭:** 중심선 각 점에서 법선 방향으로 벽까지 거리를 재서 `w_left`/`w_right`를 만듭니다. 막다른 곳이나 오목한 곳은 이웃 값으로 매끈하게 해서 "통로 폭"을 따로 둡니다.
6. **레이스 라인:** `min_curvature_line` → `refine_min_time`(생성 트랙과 같은 설정)
7. **저장:** 정방향과 역방향 `.npz`를 `installation/maps_real/`에 둡니다. 학습 목록(`list_variants`)에는 넣지 않습니다.
8. **그림 확인:** 격자, 중심선, 폭, 레이스 라인을 겹친 png를 만듭니다.

필요한 패키지: numpy, scipy (`ndimage.distance_transform_edt`), scikit-image (`morphology.skeletonize`), pyyaml, Pillow.

### 단계 C. 검증과 평가 연결
1. 변환한 트랙에서 expert를 달려 봅니다. 2랩을 완주해야 하고, 막힌 곳이 없어야 합니다. 랩 타임이 실차 기록과 대략 맞는지도 봅니다.
2. 실차 로그 궤적을 같은 지도에 겹쳐 봅니다(`log_map.py` 대신 SLAM 지도 좌표 사용).
3. `compare_models.py`에 `--extra_tracks installation/maps_real/*.npz` 같은 "실제 코스" 묶음을 추가합니다(학습에는 안 씀).

### 대안 (SLAM을 쓸 수 없을 때)
- **기존 로그를 오프라인 포즈 그래프로 다시 맞추기:** `log_map.py`에 루프 클로저를 추가하는 방식입니다. 차 없이 할 수 있지만, SLAM보다 덜 튼튼합니다.
- **줄자로 코스 그리기:** 정밀도가 낮고 시간이 많이 듭니다.

---

## 5. 순서와 시간 (예상)

| 순서 | 할 일 | 어디서 | 시간 |
|---|---|---|---|
| 1 | LiDAR 수평 확인, slam_toolbox 설치, `config/slam.yaml` 추가 | 차 | 30분 |
| 2 | 지도 세션 (1~2바퀴 + bag) + 줄자 확인 | 차 | 30분 |
| 3 | `tools/map_to_track.py` 작성 + 확인 그림 | PC (Claude) | 2~3시간 |
| 4 | expert 주행 검증, `compare_models.py` 연결 | PC (Claude) | 1시간 |

**결정 대기:** 언제 시작할지(1~2는 차가 필요), 연습 코스부터 할지 대회 코스 공개 후 할지.
