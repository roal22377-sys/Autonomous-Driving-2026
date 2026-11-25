# 실제 코스 지도 따기 → 시뮬레이터 트랙 → 평가·학습 (사용 설명서, 11/25)

차가 **스스로 코스를 천천히 몇 바퀴 돌며 LiDAR를 기록**합니다. PC에서 **루프 클로저 지도**를 만들고, 그 지도를 **시뮬레이터 트랙**으로 바꿔
평가와 학습에 씁니다. 모든 명령은 저장소 루트(`~/Autonomous-Driving-2026`) 기준입니다.

```
[차]  tools/map_session.sh                   자율 주행(IL 11, 2 m/s)으로 3바퀴 이상 + bag 기록
[PC]  tools/map_session.sh --finish NAME     기록 가져오기 → tools/map_offline.py (지도) → tools/map_to_track.py (트랙)
                                             → tools/check_track.py (시뮬레이터에서 expert·모델 주행 확인)
[PC]  (필요하면) tools/map_session.sh --rebuild maps/<run> NAME --block=x0,y0,x1,y1   코스가 아닌 곳 막기
[PC]  compare_models.py --extra_tracks ... / train_ppo_s2r.py --real_tracks ...       평가·학습에 연결
```

---

## 1. 11/24 로그로 검증한 결과

11/24 15:21 주행(IL 11, 상한 4 m/s, 62 s, 약 5바퀴)의 기록만으로 같은 흐름을 돌렸습니다.

| 단계 | 결과 | 파일 |
|---|---|---|
| 지도 (`map_offline.py`) | 키프레임 943개, 스캔 매칭 실패 52개(→ odom으로 대체), **루프 클로저 176개**. odom만 쓰면 지도가 꽃잎 모양으로 뭉개지고, 루프 클로저가 없으면(11/24 `log_map.py`) 벽이 2~3겹이었는데, **벽이 한 겹으로 정리되고 바퀴마다 궤적이 겹침** | `maps/run_1124_1521/preview.png` (왼쪽부터 odom만 / 스캔 매칭만 / 루프 클로저), `map.pgm` |
| 트랙 (`map_to_track.py`) | 한 바퀴 34.1 m, 통로 반폭 0.5~2.2 m, 안쪽 섬을 도는 통로. 바깥 벽 너머로 보인 방 하나는 `--block=-10.5,4.75,-7,6.5`로 막음 | `src/f1tenth/installation/maps_real/Practice1124(_rev).npz`, `_check.png` |
| 주행 확인 (`check_track.py`) | expert가 양방향, 상한 8·4·2 모두 완주(1랩 9.5~15.9 s). 실차 IL 11은 상한 4에서 약 12 s/랩이었음 → 크기가 맞음 | `Practice1124_drive.png` |

**지도 주행 모델 고르기** (같은 트랙, 양방향 × 상한 2·2.5·3 × 시드 3 = 18회): `check_track.py --caps 2 2.5 3 --seeds 3`

| 모델 | 완주 | 실패 |
|---|---|---|
| expert | 18/18 | |
| **IL 11** | **18/18** | → `agent_mapping.yaml`에 사용. 11/24 실차에서도 이 코스를 접촉 없이 돎 |
| PPO 12 u190 / u250 / u272 | 17/18 각각 | u190·u272: 정방향 상한 2, 출발 3.7 s 같은 자리 / u250: 역방향 상한 3 |
| PPO 5 (지금 실차 모델) | 13/18 | 정방향 상한 2에서 0/3, 2.5와 3에서 1번씩 → **실차 저속 시험 때 주의** |

---

## 2. 목요일 순서

### 준비 (PC)
```bash
cd ~/Autonomous-Driving-2026 && git pull && git status      # 바뀐 파일 없어야 함
ssh jowonjun@10.106.0.73 true                                # 비밀번호 없이 되어야 함
```
- 지도 주행 설정은 `src/f1tenth_ros/f1tenth_stack/config/agent_mapping.yaml`(IL 11, 2.0 m/s)입니다. 2.0 m/s는 학습한 가장 낮은 상한이고, 그보다 낮으면 `agent_node`가 학습 범위 밖이라고 경고합니다.
- LiDAR 수평을 확인합니다(11/24 오전처럼 기울면 바닥이 벽으로 찍힘, `CAR_TESTS_1124.md` 시험 0).

### 차에서 지도 주행
```bash
tools/map_session.sh          # git push + 차 git pull, 탭 6개: bringup / TELEOP(s=정지, r=주행) / bag / status / cmd / agent
```
1. 차를 코스의 직선 구간 가운데에 진행 방향으로 놓습니다.
2. TELEOP 탭에서 `s`를 누르고, agent 탭에서 Enter, 다시 TELEOP 탭에서 `r`을 누르면 출발합니다.
3. **3바퀴 이상** 돌게 둡니다. 같은 곳을 여러 번 지날수록 루프 클로저가 많아집니다. 언제든 `s`로 정지할 수 있습니다.
   - 벽에 닿거나 멈추면 `s` → 손으로 옮기기 → `r`. 기록은 계속됩니다. 손으로 옮기는 구간은 지도에서 빼는 게 좋습니다(아래 `--t0 --t1`).
   - 갈림길이나 막다른 곳이 있는 코스면, 자율 주행 뒤 키보드로 한 번 들어갔다 나와도 됩니다.
4. 끝나면 TELEOP 탭에서 `s`를 누릅니다.

### PC에서 지도와 트랙 만들기
```bash
tools/map_session.sh --finish Practice1127          # NAME: 영문·숫자만 (_ 없이)
```
- 하는 일: 정지 → bag 가져오기(`logs/run_MMDD_HHMM_extracted`) → `eval_drive.py` → `map_offline.py` → `maps/run_MMDD_HHMM/` → `map_to_track.py` → `src/f1tenth/installation/maps_real/Practice1127(_rev).npz` → `check_track.py` 순서입니다.
- 손으로 옮긴 구간을 빼려면 자율 주행 구간만 씁니다. 시각은 `eval.txt`의 start/dur로 압니다.
  ```bash
  tools/map_session.sh --finish Practice1127 --t0 12 --t1 140
  ```
- 다른 PC에서 기록만 받아 둔 경우는 `python3 tools/map_offline.py logs/<run>_extracted maps/<run>` 다음에 `tools/map_session.sh --rebuild maps/<run> NAME`을 실행합니다.

### 결과 확인 (3가지)
1. `maps/<run>/preview.png`: 오른쪽 패널의 **벽이 한 겹**인지, 빨간 궤적이 바퀴마다 겹치는지 봅니다. `report.txt`의 루프 클로저가 수십 개 이상이고 "fell back to odom"이 키프레임의 10 % 이하이면 정상입니다.
2. `maps_real/<NAME>_check.png`: 흰 영역이 **코스 통로만**인지, 중심선(파랑)과 레이스 라인(빨강)이 통로 안에 있는지 봅니다.
   - 방이나 열린 곳이 붙어 있으면 그 사각형 좌표를 그림에서 읽고 다시 만듭니다.
     ```bash
     tools/map_session.sh --rebuild maps/<run> Practice1127 --block=-10.5,4.75,-7,6.5
     ```
     `--block`은 여러 번 쓸 수 있습니다. 음수로 시작하는 좌표는 `=`로 붙여 씁니다.
3. `check_track.py` 출력: **expert가 모든 상한, 양방향에서 완주**하면 쓸 수 있는 트랙입니다.

### 줄자 확인 (정확도)
통로 폭 3~4곳과 긴 직선 하나를 줄자로 재서 `_check.png`의 폭과 비교합니다(목표 오차 ±5 cm). 차이가 크면 결과와 함께 알려 주세요.

### 저장
```bash
git add maps/<run> src/f1tenth/installation/maps_real/Practice1127*
git add -f logs/<run>_extracted          # logs/는 .gitignore에 있음
git commit -m "map: practice course 11/27" && git push
```

---

## 3. 평가와 학습에 쓰기

| 용도 | 명령 | 비고 |
|---|---|---|
| 모델 비교에 실제 코스 열 추가 | (src/f1tenth에서) `python -u compare_models.py ppo_s2r/5 ppo_s2r/12190 --seeds 2 --extra_tracks installation/maps_real/Practice1127.npz installation/maps_real/Practice1127_rev.npz` | 상한 8·4·2, 출발선에서 2랩. "real course" 열과 상한 8의 2랩 시간 |
| 실차 전에 한 트랙만 빨리 확인 | `python3 tools/check_track.py src/f1tenth/installation/maps_real/Practice1127.npz --models ppo_s2r/12190 --caps 2 3 4 --seeds 3` | 상한마다 완주 수, 2랩 시간, 실패 시각 |
| PPO 평가마다 실제 코스 보고 | `train_ppo_s2r.py ... --real_tracks installation/maps_real/Practice1127.npz installation/maps_real/Practice1127_rev.npz` | 저장 기준에는 안 들어가고 기록만 합니다(`[real course @ update N]`, `train_log.pkl`의 `real_success`) |
| **그 코스로 학습** (대회 코스용) | 위 옵션에 `--real_train_share 0.1` 추가 | 에피소드의 10 %를 그 코스에서 학습합니다. 연습 코스는 대회 코스와 다르니 **평가만** 하고, 대회 코스를 따면 이 옵션으로 짧게 미세조정합니다 |

- 실제 코스 트랙은 `installation/maps_real/`에 따로 둡니다. 생성 트랙 학습 목록(`maps_gen`)에 섞이지 않아서, 옵션을 주지 않으면 학습에 쓰이지 않습니다.
- 트랙 파일에 통로 폭(`w_left`/`w_right`)이 있어서 막다른 공간 벌점(`--bay_cost`)도 그대로 작동합니다.

---

## 4. 다음 단계: 지도 기반 주행 (결정 대기)

지도와 차의 위치를 알면 LiDAR 10 m 너머의 코스를 미리 알 수 있습니다. 그러면 직선에서 더 빨리 달리고 늦게 제동할 수 있습니다. 11/25 분석에서 랩 손해의 70 %가 직선 최고 속도였습니다(PPO 5.3 m/s, expert 7.3 m/s).

| 필요한 것 | 내용 |
|---|---|
| 실차 위치 추정 | 이 지도 위에서 파티클 필터(AMCL, `nav2_amcl`)나 slam_toolbox 위치 추정 모드를 씁니다. 40 Hz에 오차 5~10 cm가 목표입니다 |
| 정책 입력 추가 | 위치 좌표 그대로가 아니라 **앞 코스 정보**를 넣습니다. 앞 5·10·20·30 m의 곡률, 다음 코너까지 거리, 레이스 라인에서 벗어난 거리와 방향 오차입니다. 트랙마다 의미가 같아 일반화됩니다 |
| 시뮬레이터 | 같은 정보를 위치 추정 오차(잡음, 가끔 튐)와 함께 줍니다 |
| 재학습 | 관측이 바뀌므로 IL부터 다시 하고(`obs_dim` 증가), 그다음 PPO를 합니다 |
| 대비책 | 위치 추정이 불확실하면 지금 같은 LiDAR만 쓰는 정책으로 전환합니다 |
| 다른 선택지 | 위치 추정만 되면 expert(레이스 라인 추종)를 실차에서 그대로 돌릴 수 있습니다. 시뮬레이터에서는 완주 100 %에 PPO 12보다 약 27 % 빠릅니다 |

대회에서 미리 지도를 딸 수 있는지(규정) 확인한 뒤 결정합니다.

---

## 5. 도구 설명

| 도구 | 하는 일 |
|---|---|
| `tools/map_session.sh` | 차 세션(= `MODEL=mapping tools/drive_session.sh`). `--finish NAME`은 기록 → 지도 → 트랙 → 확인까지, `--rebuild`는 트랙만 다시 만듭니다 |
| `tools/map_offline.py LOG OUT [--t0 --t1 --res --loop_dist --loop_gap]` | 오프라인 2D SLAM입니다.<br>① 키프레임(0.15 m / 5°)마다 1081개 빔을 점-선 ICP로 주변 지도에 맞춤(odom을 초기값으로)<br>② 같은 곳을 15 s 이상 뒤에 다시 지나면 두 스캔을 맞춰 "같은 위치" 조건 추가(잔차 3 cm 미만만)<br>③ 포즈 그래프 최소제곱(scipy)으로 전체 궤적 보정, ②~③을 2번<br>④ 광선 추적으로 점유 격자를 만들어 `map.pgm`/`map.yaml`(ROS map_server 형식)로 저장. `poses.csv`, `preview.png`, `report.txt`도 남김 |
| `tools/map_to_track.py MAP --name NAME [--block --max_half 2.5 --close_gaps 0.3]` | 지도와 궤적으로 트랙을 만듭니다.<br>① 궤적과 이어진 빈 칸(경로에서 2.5 m 안, 벽의 0.6 m 이하 틈은 닫음)<br>② 한 바퀴 궤적을 좌우 벽 사이 가운데로 4번 다시 맞춘 중심선(0.1 m 간격)<br>③ 좌우 폭(1.5 m 중앙값 + 0.2 m 넘는 오목한 곳은 통로 폭에서 제외)<br>④ 생성 트랙과 같은 레이스 라인 최적화<br>⑤ 정방향과 역방향 `.npz`(`utils/track_gen.to_track` 형식) |
| `tools/check_track.py TRACK [--models --caps --seeds --workers]` | 시뮬레이터에서 expert와 모델을 그 트랙에서 달려 완주 수, 2랩 시간, 실패 시각을 보여 주고 `_drive.png`를 만듭니다 |
| `tools/log_map.py` | (분석용) 사고 구간 지도. 루프 클로저가 없어서 트랙으로는 쓰지 않습니다 |

**트랙 파일 형식**(`utils/track_gen.py`와 같음): `occupancy`(행 0 = 최소 y, >128 빈 칸), `resolution`, `origin`, `centerline`(0.1 m 간격, 주행 방향),
`w_left`/`w_right`, `raceline`, `length`, `runoffs`(실제 코스는 빈 배열), `name`, `map`, `reverse`, `original=True`, `gen_sig='real:<NAME>'`.

**선택: 차에서 실시간 지도 (slam_toolbox).** 위 흐름에는 필요 없습니다. 원하면 차에 `sudo apt install ros-foxy-slam-toolbox ros-foxy-nav2-map-server`를 설치하고, bringup 중에 `ros2 launch slam_toolbox online_async_launch.py`(인자 이름은 Foxy 버전에서 확인)를 실행한 뒤 `ros2 run nav2_map_server map_saver_cli -f ~/maps/<name>`으로 저장합니다.
이 경우 궤적(`poses.csv`)이 따로 없습니다. 그래서 `map_to_track.py`에는 같은 기록으로 `map_offline.py`를 돌려 얻은 `poses.csv`를 `--poses`로 줍니다. 두 지도는 좌표계가 다르므로 `map_offline.py` 지도를 쓰는 쪽을 권합니다.
