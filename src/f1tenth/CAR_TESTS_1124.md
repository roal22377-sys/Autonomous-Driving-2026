# 실차 시험 계획 (2026-11-24): 실차와 시뮬레이터의 약 30 % 차이 찾기

IL 11 실차 주행 로그(`logs/run_1124_1344_extracted`)를 분석한 결과와, 그 원인을 확정하기 위한 실차 시험 순서입니다.
어느 로컬 PC에서든 이 문서대로 하면 됩니다. 명령은 모두 **저장소 루트**(`~/Autonomous-Driving-2026`)에서 실행합니다.

---

## 1. 11/24 IL 11 주행에서 찾은 것

### 무엇이 문제였나
| 항목 | 값 |
|---|---|
| 자율 주행 4구간 | **3구간 벽 접촉** (4~9초 만에), 속도 상한 3 m/s, 평균 2.1 m/s |
| 흔들림 | **1.9 Hz로 좌우 진동**, 사고 직전 3초 동안 0.5초마다 최대 조향의 ±50 %씩 번갈아 꺾음 |
| "넓은 공간으로 가는" 현상 | 흔들리는 동안 차 방향이 넓은 쪽을 향하는 순간 그쪽으로 들어가 벽에 닿음. 넓은 공간은 코스 안쪽 |
| LiDAR, 명령 주기 | 정상 (사고 전 빠진 빔 거의 없음, 40 Hz로 시뮬레이터와 같음) |

### 원인 후보: 실차와 시뮬레이터의 반응이 약 30 % 어긋남
명령한 조향에 비해 실제로 얼마나 도는지(실제 요레이트 ÷ `v·tan(조향)/0.34`)를 로그와 시뮬레이터에서 같은 방법으로 잼:

| | 조향 지연 | 왼쪽 | 오른쪽 |
|---|---|---|---|
| 실차 (odom 속도 기준) | 0.10 s | **0.72** | **0.86** |
| 시뮬레이터 (IL 11, 상한 3 m/s) | 0.125 s | 0.52 | 0.52 |

두 가지 설명이 가능하고, **둘 중 어느 쪽인지가 시험의 목적**입니다.

| 가설 | 내용 | 맞다면 생기는 일 |
|---|---|---|
| **A. 조향 응답 차이** | 실차가 같은 명령에 시뮬레이터보다 40~65 % 더 돈다 (시뮬레이터 타이어 접지 μ 0.5가 너무 낮게 맞춰졌을 수 있음) | 정책이 과하게 돌고 → 반대로 고치고 → 진동. 지금 흔들림과 맞음 |
| **B. 속도 측정 차이** | 스캔 매칭으로 잰 실제 속도가 odom보다 약 **1.3배** 빠름 → `speed_to_erpm_gain`(지금 4180)이 틀렸을 수 있음 | 차가 생각보다 30 % 빨리 달림. 정책이 자기 속도를 낮게 앎. 11/23 접지 측정도 odom 속도를 썼다면 같이 틀어짐 |

분석 그림: `logs/run_1124_1344_extracted/analysis/` (구간별 지도, 사고 직전 LiDAR 장면)

---

## 2. 시작 전 준비 (로컬 PC)

```bash
cd ~/Autonomous-Driving-2026
git pull                 # drive_session.sh는 시작할 때 이 PC에서 push하고 차가 pull함 → 최신이어야 함
git status               # 바뀐 파일이 없어야 함 (-m 옵션은 이 PC의 변경을 전부 커밋함)
ssh jowonjun@10.106.0.73 true    # 비밀번호 없이 되어야 함 (같은 Wi-Fi)
```

- `drive_session.sh`가 탭을 열고 차에서 bringup, 키보드 조종(비상 정지), 기록을 실행합니다.
- **키보드 탭에서 `s` = 정지, `r` = 주행**입니다. 시험 중에는 항상 키보드 탭을 앞에 두세요.

---

## 3. 시험 순서

### 시험 1. 직선 속도 측정 (약 15분) ⭐ 가장 중요
```bash
tools/drive_session.sh --speed 2 2.5 3       # 마지막 탭에서 속도마다 계획 → Enter = 출발, s = 건너뜀, q = 끝
tools/drive_session.sh --finish --speed      # 종료 + 기록 가져오기 + 분석 (tools/speed_analysis.py)
```
- **공간**: 평평한 벽을 향한 직선 6~10 m (3 m/s 기준). 차는 벽 앞에서 스스로 제동합니다.
- **알 수 있는 것**: LiDAR로 벽까지 거리가 줄어드는 속도 = 실제 속도 → odom과 비교, `speed_to_erpm_gain` 보정값, 가속·제동(m/s²)
- **판단**: 실제 속도가 odom보다 약 1.3배 → **가설 B**. 거의 같음(±5 %) → **가설 A**.

### 시험 2. 원선회 (약 20분)
```bash
tools/drive_session.sh --circle              # 조향 0.44, -0.50(양쪽 최대), 0.3, -0.3, 0.15, -0.15 차례로
tools/drive_session.sh --finish --circle     # 종료 + 분석 (tools/circle_analysis.py)
```
- **공간**: 지름 4~5 m 빈 바닥. 조향마다 차를 가운데 두고 키보드 탭 `r` → 마지막 탭 Enter.
- **알 수 있는 것**: 조향 명령별 실제 회전(왼쪽/오른쪽 따로), 타이어 접지 μ. 시험 1의 실제 속도와 함께 보면 가설 A의 크기를 정확히 알 수 있음.

### 시험 3. 진단 주행 (각 5~10분, 같은 코스)
| 주행 | 설정 (`src/f1tenth_ros/f1tenth_stack/config/agent_il.yaml`) | 보는 것 |
|---|---|---|
| 3-A | `speed_limit: 2.0` (나머지 그대로) | 느리면 흔들림·충돌이 줄어드는지 (줄면 속도·접지 쪽) |
| 3-B | 시험 1·2 결과로 계산한 `steer_gain_left/right` (Claude가 계산해서 알려줌) | 실차 조향 응답을 시뮬레이터에 맞추면 흔들림이 사라지는지 → 원인 확정 |

```bash
MODEL=il tools/drive_session.sh              # 주행 (agent 탭에서 Enter, 키보드 탭 r)
tools/drive_session.sh --finish              # 종료 + 기록 + 평가 (tools/eval_drive.py)
```
- 3-B는 **시험 1·2 결과를 본 뒤에** 합니다. 지금 추정값을 넣으면, 원인이 가설 B일 때 오히려 덜 꺾어서 위험합니다.
- yaml 값을 바꾼 뒤 `drive_session.sh`를 실행하면 커밋 없이 차로 복사되지 않을 수 있으니, 바꾼 뒤 `git commit`과 `git push`를 하고 실행하거나 `-m "메시지"` 옵션을 쓰세요.

---

## 4. 결과 보내기

각 시험의 `--finish` 뒤:
```bash
git add -f logs/<run>_extracted && git commit -m "car test 11/24: <시험 이름>" && git push
```
- `logs/`는 용량 때문에 Git에서 제외되어 있어서 `-f`가 필요합니다.
- 터미널에 나온 분석 출력을 Claude에게 붙여 넣어도 됩니다. **시험 1만 해도 원인의 절반은 결정됩니다.**

## 5. 로그를 직접 분석하려면 (선택)

| 도구 | 하는 일 |
|---|---|
| `python3 tools/eval_drive.py logs/<run>_extracted` | 벽 접촉, 흔들림, 속도 요약 (`--finish`가 자동 실행) |
| `python3 tools/log_map.py logs/<run>_extracted out.png <시작 s> <끝 s>` | LiDAR 스캔 매칭으로 구간 지도 + 궤적 + 조향 |
| `python3 tools/log_view.py logs/<run>_extracted <출력 폴더>` | 자율 구간마다 사고 0.5~2초 전 LiDAR 장면 |
| `python3 tools/steer_response.py logs/<run>_extracted <시작> <끝> ...` | 실차 조향 지연과 좌우 응답(위 표의 값), 스캔 매칭 속도 / odom 속도 |

자율 구간의 시작·끝 시각은 `eval_drive.py` 출력의 `start`와 `dur`로 알 수 있습니다(사고 전까지만 넣기).
필요한 패키지: numpy, pandas, scipy, matplotlib.

## 6. 결과가 나오면

| 결과 | 다음 일 |
|---|---|
| 가설 B (odom이 틀림) | `config/vesc.yaml`의 `speed_to_erpm_gain` 수정 → 접지 μ 다시 계산 → 시뮬레이터 보정 |
| 가설 A (조향 응답 차이) | 시뮬레이터의 타이어 접지·조향 응답을 실차 값으로 (`train_ppo_s2r.py --sim_override`) |
| 공통 | 보정된 시뮬레이터로 PPO 미세조정 (PPO 4 최고 모델에서 이어서) → 실차 IL+PPO 시험 (`MODEL=ilppo`) |
