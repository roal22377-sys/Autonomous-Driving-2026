# 11/26(목) 수업 시간 실차 계획: 지도 따기 → LiDAR PPO 주행 → 위치 확인 → 위치 정보 모델(계획 B)

## 한눈에 보기 (15:00 시작, 약 2.5시간) — 11/26 14:58 갱신

**다른 컴퓨터에서 할 때는 먼저 0절(준비, 15분)을 합니다.** 모든 명령은 저장소 루트(`~/Autonomous-Driving-2026`) 기준입니다.

### 출발 전 (PC, 5분)
```bash
cd ~/Autonomous-Driving-2026 && git pull && git status       # 바뀐 파일이 없어야 함
ssh jowonjun@10.106.0.73 true                                 # 비밀번호 없이 되어야 함 (0-2)
ssh jowonjun@10.106.0.73 'source ~/venv/torch210/bin/activate && python3 -c "import scipy; print(scipy.__version__)"'
```
- 차 IP가 바뀌었으면 모든 명령 앞에 `CAR_HOST=jowonjun@<새 IP>`를 붙입니다. 예: `CAR_HOST=jowonjun@10.106.0.80 tools/map_session.sh`
- scipy가 없으면 3번(위치 정보 모델) 전에 차에서 `pip install "scipy<1.11"`을 실행합니다. 1·2번에는 필요 없습니다.
- LiDAR 수평과 배터리 완충을 확인합니다. **줄자**를 챙깁니다(통로 폭 측정).

### 할 일

| # | 할 일 | 시간 | 필수 여부 | 명령 |
|---|---|---|---|---|
| 1 | **지도 따기** | 30분 | 필수 | `tools/map_session.sh` → TELEOP `s` → agent Enter → TELEOP `r` → 3바퀴 → `s` → `tools/map_session.sh --finish Practice1126` → 확인 4가지(2절) → commit / push |
| 2 | **LiDAR PPO 주행** | 40분 | 필수 | `MODEL=ilppo tools/drive_session.sh` 상한 2 → 4 → 5, 괜찮으면 `MODEL=ilppo_fast` 5 → 6. 상한마다 3바퀴, 끝날 때마다 `tools/drive_session.sh --finish` |
| 3 | **위치 확인** | 15분 | 필수(4번 전) | 차를 출발 자리에 놓고 `MODEL=preview tools/drive_session.sh` + **다른 터미널에서 `MODEL=preview tools/drive_session.sh --loc`** → **OK** 확인 → 손으로 밀어 한 바퀴(계속 OK) (4절) |
| 4 | **위치 정보 모델** | 40분 | 조건부(3번 통과) | 같은 세션에서 TELEOP `r`. 상한 2.5 → 3 → 4 → 5. `--loc` 화면은 계속 켜 둠. 벽에 닿으면 `MODEL=preview_robust`, 시간이 남으면 `MODEL=preview_il` 비교 |
| 5 | 원선회 | 10분 | 선택 | 공간이 있으면 `tools/drive_session.sh --circle`(조향 양 끝 2개만) |

- **2번 기록 하나로 세 가지를 합니다:** 실차 결과, 접지력 측정(5-1절 `grip_from_log.py`), 위치 추정 시험(4-1절 `loc_test.py`).
- **시간이 부족하면** 1 → 2 → 3 → 4 순서로 하고, 5번과 상한 6은 생략합니다.
- **위치 정보 모델은 위치를 0.3 s 넘게 잃으면 스스로 멈춥니다**(정상 동작). 다시 찾으면 이어서 달립니다.
- **끝나면:** `git add -f logs/run_1126_*_extracted && git commit -m "car 11/26 logs" && git push`. 본 것(벽 접촉, 흔들림, 좋았던 주행, 줄자 값)은 메시지로 남겨 주세요. 다음 학습은 이 기록을 보고 같이 정합니다(8절).

### 위치가 맞는지 보는 법 (3·4번, `tools/loc_view.py`)

`MODEL=preview tools/drive_session.sh --loc`를 PC의 **다른 터미널**에서 켭니다. agent가 시작 위치를 찾으면 화면이 나옵니다. 차에 아무 명령도 보내지 않습니다(읽기만 함).

| 판정 | 뜻 | 할 일 |
|---|---|---|
| **OK** (초록) | 스캔의 75 % 이상이 지도 벽 위에 있음 (맞는 위치는 보통 95–100 %) | 출발해도 됨 |
| **CHECK** (노랑) | 45–75 %. 차 근처에 사람이나 물건이 있어도 이렇게 나옴 | 그림을 보고 판단. 사람을 비키게 하고 다시 봄 |
| **WRONG** (빨강) | 45 % 미만: 위치가 틀림 | 아래 "다시 잡기" |
| **LOST** | agent가 위치를 잃었다고 보고함 | 아래 "다시 잡기" |

- **그림:** `.` 지도 벽, `o` 벽 위에 맞은 스캔, `X` 벽에 안 맞은 스캔, `@` 차, `*` 차가 보는 방향. 맞으면 `o`가 `.` 줄 위에 겹칩니다.
- **RViz로 보기(선택):** PC에 ROS 2 Foxy가 있고 차와 같은 네트워크에서 토픽이 보이면 다음을 켭니다.
  ```bash
  source /opt/ros/foxy/setup.bash && export ROS_DOMAIN_ID=6 && rviz2 -d tools/loc_view.rviz
  ```
  초록 점(지금 스캔)이 검은 지도 벽 위에 겹치면 맞습니다. 빨간 화살표가 차입니다. `--loc` 터미널이 켜져 있어야 지도와 초록 점이 나옵니다.
  화면이 비어 있으면 `ros2 topic list`에 `/loc_map`이 없는 것입니다(WSL은 기본 설정에서 차의 토픽이 안 보임). 이때는 터미널 판정을 씁니다. 기준은 같습니다.
- **다시 잡기:** TELEOP `s` → agent 탭 Ctrl-C → 차를 출발 자리에 놓기(코너가 보이는 곳, 통로에 사람 없음) → `MODEL=preview tools/drive_session.sh --agent` → `--loc` 화면이 OK가 된 뒤 TELEOP `r`.
- **차를 손으로 옮겼으면** 항상 다시 잡습니다(agent가 들고 다니는 중의 위치를 믿을 수 있게 하려면 출발 자리에서 켜는 게 확실합니다).

### 주의: 좁은 헤어핀 (11/26 14:56 시뮬레이터 결과, `results/reports/hall_eval_1126_1456.md`)

2025 대회 트랙 사진을 보고 만든 좁은 코스(통로 1.0–2.0 m, 벽 하나 사이로 붙은 헤어핀, 평가 전용 53개)에서는 **지금 모델이 거의 완주하지 못합니다.** expert는 53/53입니다.

| 모델 | 상한 8 | 상한 4 | 2025 복제 맵 |
|---|---|---|---|
| `preview` PPO 21 u30 | 0 % | 2 % | 0/4 |
| `preview_robust` PPO 21 u50 | 0 % | 0 % | 0/4 |
| `preview_il` IL 13 | 0 % | 0 % | 0/4 |
| `ilppo` PPO 12 u190 (LiDAR 전용) | 19 % | 23 % | 0/4 |

- 위치 정보 모델은 대부분 출발 4–5 s 안(첫 좁은 코너)에서 부딪힙니다. 2025 복제 맵에서는 상한 2.5에서도 1.6 m 간격 헤어핀 두 개 구간에서 부딪혔습니다.
- 11/24 연습 코스(가장 좁은 곳 약 1.0 m)에서는 세 위치 정보 모델 모두 상한 2–4에서 완주했으므로, 오늘 주행 계획은 그대로 갑니다.
- 오늘 연습 코스에 폭 1.5 m 이하 통로나 헤어핀이 있으면, 그곳은 **상한 2–2.5에서 먼저** 봅니다.
- 그런 곳의 실차 기록(벽 접촉 위치, 속도)이 다음 학습(좁은 코스 학습)에 가장 중요한 자료입니다. 어디서 닿았는지 적어 주세요.
- 전체 결과는 `results/reports/hall_eval_1126_1456.md`에 있습니다(8절).

### 확정 모델 (3시드 비교 `results/reports/compare_1126_0837_*.md`, `_1056_*`, `_1330_*`)

| 설정 (`MODEL=`) | 모델 | 용도 | 짧은 코스+연습 코스 | 상한 8 2랩 | 최악 조건 | 막다른 공간 | 일반 |
|---|---|---|---|---|---|---|---|
| `mapping` (`map_session.sh`) | IL 11 | 지도 주행 2 m/s | – | – | – | – | – |
| `ilppo` | PPO 12 u190 | LiDAR 전용, 안정 | – | – | – | – | – |
| `ilppo_fast` | PPO 12 u272 | LiDAR 전용, 약 12 % 빠름 | – | – | – | – | – |
| **`preview`** | **PPO 21 u30** (`ppo_s2r/2130`) | **위치 정보, 기본** | **99.4 %** | **30.9 s** | 82.6 % | 94.3 % | 99.2 % |
| `preview_robust` | PPO 21 u50 (`ppo_s2r/2150`) | 위치 정보, 강건 | 98.8 % | 31.1 s | **86.4 %** | 93.6 % | 99.2 % |
| `preview_il` | IL 13 (`imitation/13`) | 위치 정보, 비교용 | 97.5 % | 31.0 s | 78.0 % | 94.3 % | **100 %** |

- 위치 정보 모델 세 개 모두 연습 코스 양방향, 상한 2 / 2.5 / 3 / 4에서 3/3 완주했습니다.
- **상한 8(제한 없음) 역방향에서는 세 모델 모두 위쪽 돌출부(지도 x ≈ −3.4) 부근에서 가끔 닿았습니다.** 오늘은 상한 5까지만 씁니다.
- 6개 설정 모두 11/24 실제 기록으로 차 노드 재생 시험을 통과했습니다(`tools/agent_replay.py`, 14:00).
- PC에서는 PPO 24(PPO 21 u30을 조심스럽게 다듬는 학습)가 시험 중에도 계속 돕니다. 오늘 차 설정과는 무관합니다.

목표는 네 가지입니다.
1. 연습 코스 지도를 따서 맵 툴을 실차로 검증합니다.
2. 지금까지 가장 좋은 LiDAR 전용 PPO의 실차 결과를 얻습니다. 같은 코스를 시뮬레이터에도 넣어 랩 타임을 직접 비교합니다.
3. 그 지도 위에서 위치 추정이 되는지 확인합니다.
4. 준비되면 위치 정보 입력 모델(계획 B)을 낮은 속도부터 돌려 봅니다.

연습 코스는 대회 코스와 무관합니다. 대회 코스는 대회 이틀 전에 땁니다. 오늘은 그날의 흐름을 한 번 해 보는 날입니다.

---

## 0. 다른 컴퓨터에서 할 때 (처음 한 번, 15분)

수업 시간에는 이 PC가 아니라 다른 컴퓨터로 해도 됩니다. 필요한 것은 **Linux 셸(Ubuntu, WSL, macOS), git, ssh, Python 3.8 이상**입니다.

**0-1. 저장소**
```bash
git clone https://github.com/roal22377-sys/Autonomous-Driving-2026.git ~/Autonomous-Driving-2026
cd ~/Autonomous-Driving-2026 && git config user.name "<이름>" && git config user.email "<메일>"
```
- 지도와 주행 기록을 push해야 하므로 이 컴퓨터에서 GitHub에 로그인돼 있어야 합니다(`git push`가 되는지 먼저 확인).

**0-2. 차에 비밀번호 없이 접속**
```bash
ssh-keygen -t ed25519            # 키가 이미 있으면 생략 (엔터만 눌러도 됨)
ssh-copy-id jowonjun@10.106.0.73
ssh jowonjun@10.106.0.73 true    # 비밀번호를 묻지 않으면 성공
```
- 스크립트가 ssh를 여러 번 부르므로 비밀번호를 물으면 진행이 안 됩니다.

**0-3. PC 쪽 Python (지도 만들기, 주행 평가, 시뮬레이터 확인)**
```bash
tools/setup_train_env.sh         # ~/venv/f1tenth 에 고정 버전 설치 (Python 3.8 기준, 약 10분)
```
- 스크립트들은 `~/venv/torch210` → `~/venv/f1tenth` → `python3` 순서로 Python을 찾습니다. 다른 곳에 설치했으면 `PY=/경로/python tools/map_session.sh --finish ...`처럼 붙입니다.
- 시간이 없으면 최소한 `pip install numpy scipy matplotlib pandas pyyaml numba`만 해도 **지도 만들기(`map_session.sh --finish`)와 주행 평가는 됩니다.** 마지막 시뮬레이터 확인(`check_track.py`)만 실패하는데, 그 전에 지도와 차 설정은 이미 저장됩니다. 확인은 push 뒤에 Claude가 이 PC에서 합니다. (11/26 확인: 이 6개만 깐 새 환경에서 11/24 지도 → 코스 변환이 됨)

**0-4. 탭 6개 열기**
- `drive_session.sh` / `map_session.sh`는 gnome-terminal로 탭 6개를 엽니다(Ubuntu 데스크톱).
- gnome-terminal이 없으면(macOS, WSL 등) 스크립트가 **터미널 6개에서 실행할 명령 6줄**을 출력하고 끝납니다. 터미널을 6개 열어 1번(bringup)부터 순서대로 한 줄씩 실행합니다. bringup이 뜬 뒤 나머지를 켭니다.

**0-5. RViz (선택)**
- Ubuntu 20.04에 ROS 2 Foxy(`ros-foxy-desktop`)가 있으면 위치 확인을 RViz로도 볼 수 있습니다(위 "위치가 맞는지 보는 법").
- 없으면 터미널 판정(`--loc`)만으로 충분합니다.

---

## 1. 시작 전 (PC, 10분)

```bash
cd ~/Autonomous-Driving-2026 && git pull && git status          # 바뀐 파일이 없어야 함
ssh jowonjun@10.106.0.73 true                                    # 비밀번호 없이 되어야 함
ssh jowonjun@10.106.0.73 'source ~/venv/torch210/bin/activate && python3 -c "import scipy; print(scipy.__version__)"'
```
- **scipy가 없으면** 차에서 `pip install "scipy<1.11"`을 실행합니다. 위치 정보 모델(`MODEL=preview`)에만 필요합니다.
  - LiDAR 전용 모델과 지도 주행은 scipy와 numba 없이 됩니다(11/25 저녁에 확인).
  - 설치 뒤 agent_node가 쓰는 방식으로 다시 확인합니다.
    ```bash
    ssh jowonjun@10.106.0.73 'export PYTHONPATH=$HOME/venv/torch210/lib/python3.8/site-packages:$PYTHONPATH && /usr/bin/python3 -c "from scipy.spatial import cKDTree; print(1)"'
    ```
  - 설치하려면 차가 인터넷에 연결돼 있어야 합니다(`ping -c 2 pypi.org`).
- LiDAR 수평을 확인합니다(`CAR_TESTS_1124.md` 시험 0). 배터리는 완충합니다.

---

## 2. 지도 주행 (30분) — 가장 먼저

자세한 내용은 `MAP_TOOL.md` 2절에 있습니다.

```bash
tools/map_session.sh                      # 탭 6개. TELEOP s → agent 탭 Enter → TELEOP r → 출발 (IL 11, 2 m/s)
# 3바퀴 이상 돈 뒤 TELEOP s
tools/map_session.sh --finish Practice1126
```

확인할 것:
1. `maps/run_1126_*/preview.png`의 오른쪽 패널에서 벽이 한 겹이어야 합니다. `report.txt`의 루프 클로저는 수십 개 이상, "fell back to odom"은 키프레임의 10 % 이하가 정상입니다.
2. `maps_real/Practice1126_check.png`에서 흰 영역이 코스 통로만이어야 합니다. 아니면 `--rebuild ... --block=...`으로 다시 만듭니다.
3. `check_track.py` 출력에서 expert가 모든 상한, 양방향에서 완주해야 합니다.
4. **줄자:** 통로 폭 3–4곳과 긴 직선 하나를 잽니다. 목표 오차는 ±5 cm입니다. 잰 값과 `_check.png`의 값을 같이 보내 주세요.

`--finish`는 `agent_preview.yaml`의 `course`와 `map_dir`을 이 코스로 바꿔 둡니다. 저장은 다음처럼 합니다(차가 지도를 읽으려면 push가 필요).
```bash
git add -f maps/run_1126_* src/f1tenth/installation/maps_real/Practice1126* src/f1tenth_ros/f1tenth_stack/config/agent_preview.yaml
git commit -m "map: practice course 11/26" && git push
```

---

## 3. LiDAR 전용 PPO 주행 (40분)

```bash
MODEL=ilppo tools/drive_session.sh        # PPO 12 u190 (안정). speed_limit 2.0 → 4.0 → 5.0 순서로 올리기
MODEL=ilppo_fast tools/drive_session.sh   # PPO 12 u272 (빠름). 앞 모델이 5.0까지 문제없을 때만
tools/drive_session.sh --finish           # 주행마다: 정지 + bag 가져오기 + eval_drive.py
```
- 상한을 바꿀 때는 yaml의 `speed_limit`을 고치고 agent 탭만 다시 시작합니다(`tools/drive_session.sh --agent`).
- 상한마다 **3바퀴 이상** 돕니다. 이 기록은 4번의 위치 추정 시험에도 그대로 씁니다.
- **같은 코스 시뮬레이터 비교**(PC):
  ```bash
  python3 tools/check_track.py src/f1tenth/installation/maps_real/Practice1126.npz --models ppo_s2r/12190 ppo_s2r/12272 --caps 2 4 5 --seeds 3
  ```
  실차 랩 타임(`eval.txt`)과 시뮬레이터 랩 타임의 차이가 지금까지 중 가장 정확한 sim2real 차이입니다.

---

## 4. 위치 추정 확인 (20분) — 차가 위험하지 않은 순서로

**4-1. 기록으로 (PC, 위험 없음).** 3번 주행 기록을 새 지도에서 위치 추정합니다.
```bash
python3 tools/loc_test.py logs/run_1126_XXXX_extracted maps/run_1126_YYYY --track src/f1tenth/installation/maps_real/Practice1126.npz
```
- XXXX는 3번 주행, YYYY는 2번 지도입니다.
- **기준:** `valid` 98 % 이상, 끊김 0.5 s 미만, `loc_test.png`에서 궤적이 통로 안에서 매끈해야 합니다. 4–5 m/s 주행에서도 이 기준을 확인합니다.
- `--truth maps/run_1126_YYYY/poses.csv`는 지도 주행 기록을 넣었을 때만 의미가 있습니다(같은 주행의 지도 포즈와 비교).

**4-2. 차에서 정지·손으로 밀기.**
1. **먼저 차를 코스 위 출발 자리에 놓습니다.** 그다음 `MODEL=preview tools/drive_session.sh`를 실행하고, TELEOP에서 `s`를 누른 채 agent 탭에서 Enter를 누릅니다.
   - 시작 위치는 agent가 켜질 때 찾습니다. 차를 들고 다니는 중에 켜면 엉뚱한 곳을 찾을 수 있습니다. 내려놓은 뒤 다시 찾게 고쳤지만, 출발 자리에서 켜는 게 확실합니다.
   - 주행 사이에 차를 손으로 옮겼다면 agent를 다시 시작합니다(`tools/drive_session.sh --agent`).
   - 주변에 사람이 많이 서 있거나 직선 한가운데처럼 비슷한 곳이 여러 군데면 `ambiguous start` 또는 `no reliable start pose`가 나옵니다. 이때는 코너가 보이는 자리로 옮기면 됩니다.
2. PC의 다른 터미널에서 `MODEL=preview tools/drive_session.sh --loc`를 켭니다(위 "위치가 맞는지 보는 법"). **OK**가 나와야 합니다.
   - agent 로그에는 `[preview] localization: start at x .. y .. yaw ..`(시작 위치 찾기)와 5초마다 `[preview] valid ...% ... step X ms`가 나와야 합니다.
3. 차를 손으로 밀어 코스를 한 바퀴 돕니다. `--loc` 화면이 계속 OK이고, agent 로그의 `valid`가 100 % 근처를 유지해야 합니다.
4. **`step`이 10 ms를 넘으면** `agent_preview.yaml`에서 `loc_stride: 4`로 바꿉니다(Jetson이 PC보다 느림).

---

## 5. 위치 정보 모델 주행 (계획 B, 30분) — 4번이 통과했을 때만

```bash
MODEL=preview tools/drive_session.sh          # 기본: PPO 21 u30 (ppo_s2r/2130)
MODEL=preview_robust tools/drive_session.sh   # 강건: PPO 21 u50 (ppo_s2r/2150), 아래 경우에 바꿔 탐
MODEL=preview_il tools/drive_session.sh       # 비교: IL 13
```
- **강건 모델로 바꾸는 경우:** 기본 모델이 같은 상한에서 미끄러지거나 벽에 닿는 경우입니다(특히 코너 출구에서 바깥 벽).
  - PPO 21 u50은 시뮬레이터 최악 조건(지연·접지력·조향이 모두 나쁜 쪽) 완주가 86.4 %입니다(3시드). 기본 모델은 82.6 %, IL 13은 78.0 %입니다. 속도는 기본 모델보다 약 0.3 % 느립니다.
  - 두 모델 설정의 `course` / `map_dir`은 `map_session.sh`가 함께 바꿉니다.
- `speed_limit` 2.5에서 시작해 3 → 4 → 5로 올립니다. 매번 3바퀴 이상 돌고 `--finish`로 기록합니다.
- **위치를 0.3 s 넘게 잃으면 차가 스스로 멈춥니다**(속도 0, 조향은 모델 그대로). 위치를 다시 찾으면 이어서 달립니다. 로그에는 `[preview] localization lost: stopping after 0.3 s`가 나옵니다.
  - 이유: 11/25 점검에서 IL 13은 위치 없이 처음부터 끝까지 달리면 완주가 일반 트랙 9 %, 짧은 코스 53 %였습니다(위치가 있으면 93 %, 97 %). 학습 때 끊김을 0.5–3 s로 짧게만 넣었기 때문입니다.
  - PPO 14부터는 긴 끊김과 위치 없는 에피소드를 넣어 학습합니다(`configs/task/sim_override_preview_lost.yaml`).
  - `lost_stop: 0`으로 바꾸면 멈추지 않고 LiDAR만으로 계속 달립니다. PPO 14의 위치 없는 주행 성능을 보고 정합니다.
- 로그에 `[preview] car faces the other way: using ..._rev.npz`가 나오면 코스 방향을 바꾼 것입니다. 실제 주행 방향과 맞는지 확인합니다.
- 주행 평가(`eval_drive.py`)에 `localization while driving: valid ...%`가 나옵니다. 위치 추정의 실제 성적입니다.
- 같은 상한에서 3번 LiDAR 전용 모델과 랩 타임, 벽과의 거리를 비교합니다(`eval.txt`).

---

## 5-1. 접지력 측정 (원선회 공간이 없을 때: 주행 기록으로)

PPO가 느려지는 주원인은 바닥 접지력을 모르는 것입니다. 바닥은 연습장과 대회장이 같으므로, 한 번 재면 학습 범위를 좁힐 수 있습니다.
- **주행 순서:** 3번 LiDAR PPO 주행을 상한 4 → 5 → 6 순서로, 각각 3바퀴 이상 합니다.
  - 벽 접촉이나 차가 밀리는 모습이 보이면 그 상한에서 멈춥니다.
- **분석 (PC, 주행마다):**
  ```bash
  python3 tools/grip_from_log.py logs/run_1126_XXXX_extracted maps/run_1126_YYYY --track src/f1tenth/installation/maps_real/Practice1126.npz
  ```
  - 나오는 `LOWER BOUND`(충돌 없이 낸 횡가속, mu 하한)와 10초 구간별 최대 횡가속을 봅니다.
  - 상한을 올려도 같은 코너의 최대 횡가속이 더 오르지 않으면, 그 값이 접지력입니다.
- **11/24 기록 결과:** 상한 4 m/s에서 mu ≥ 0.40까지만 알 수 있었습니다(한계까지 안 감). 시뮬레이터 명목은 0.65, PPO 18은 0.55 이상을 가정합니다.
- **원선회 공간이 있으면:** `tools/drive_session.sh --circle`(조향 양 끝 2개, 약 10분)이 더 정확합니다. 타이어가 닳으니 마지막에 합니다.

## 6. 결과 보내기

- 커밋과 push: 지도(`maps/run_1126_*`), 트랙(`maps_real/Practice1126*`), 주행 기록. 기록은 `git add -f logs/run_1126_*_extracted`로 넣습니다.
- 줄자로 잰 값, 각 주행의 모델과 상한, 눈으로 본 문제(벽 접촉, 흔들림, 출발)를 적어 주세요.
- Claude가 할 일: 시뮬레이터와 실차 비교 → 시뮬레이터 보정 → 위치 추정 잡음 모델(`configs/task/f1tenth_preview.yaml` `track_preview.noise`)을 실측에 맞춤 → 다음 학습.

## 7. 대략의 시간

| 순서 | 시간 |
|---|---|
| 1 준비 | 10분 |
| 2 지도 | 30분 |
| 3 LiDAR PPO | 40분 |
| 4 위치 확인 (`--loc`) | 15–20분 |
| 5 계획 B | 30분 |
| 합계 | 약 2시간 10분 (배터리 교체 포함 2.5시간) |

## 8. 끝난 뒤: 다음 학습 방향 정하기 (Claude와 같이)

오늘 기록을 push하면 Claude가 아래를 분석해서 가져옵니다. 학습은 이 결과를 보고 같이 정한 뒤 시작합니다.

| 분석 | 무엇으로 | 나오는 것 |
|---|---|---|
| 시뮬레이터 vs 실차 랩 타임 | 2·4번 `eval.txt` + 같은 코스 `check_track.py` | sim2real 차이 (시뮬레이터 보정량) |
| 접지력 | `grip_from_log.py` (5-1절) | mu 하한 / 한계 → 학습 접지력 범위 |
| 위치 추정 정확도 | `loc_test.py`, `eval_drive.py`의 `localization while driving` | 위치 잡음 모델(`track_preview.noise`) 실측 보정 |
| 모델 비교 | 같은 상한의 LiDAR PPO / preview / preview_robust / preview_il | 어느 계열로 계속 갈지 |
| 좁은 코스 | `results/reports/hall_eval_*.md` (`tools/eval_hall.py`) + 오늘 좁은 곳 실차 기록 | 좁은 코스 학습이 필요한 정도 |

**같이 정할 후보 (Claude 의견):**
1. **좁은 코스 학습 (우선):** 2025 사진 기반 코스 99개(`utils/track_hall.py`, `installation/maps_hall`, expert 완주 확인)를 IL과 PPO 학습에 섞습니다. 지금 모델은 1.2 m 통로 헤어핀을 못 돕니다.
2. **시뮬레이터 보정:** 오늘 기록의 랩 타임·접지력·조향 반응으로 시뮬레이터 범위를 실측에 맞춥니다.
3. **위치 잡음 보정:** 실차 위치 추정 오차와 끊김을 학습의 위치 잡음에 반영합니다.
- 대회 코스는 대회 이틀 전에 지도를 따서 같은 파이프라인(지도 → 코스 → 확인 → 필요하면 추가 학습)으로 넣습니다.
