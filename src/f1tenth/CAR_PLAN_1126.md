# 11/26(목) 실차 계획: 지도 따기 → LiDAR PPO 주행 → 위치 추정 → 위치 정보 모델(계획 B)

목표는 네 가지입니다.
1. 연습 코스 지도를 따서 맵 툴을 실차로 검증합니다.
2. 지금까지 가장 좋은 LiDAR 전용 PPO의 실차 결과를 얻습니다. 같은 코스를 시뮬레이터에도 넣어 랩 타임을 직접 비교합니다.
3. 그 지도 위에서 위치 추정이 되는지 확인합니다.
4. 준비되면 위치 정보 입력 모델(계획 B)을 낮은 속도부터 돌려 봅니다.

연습 코스는 대회 코스와 무관합니다. 대회 코스는 대회 이틀 전에 땁니다. 오늘은 그날의 흐름을 한 번 해 보는 날입니다.
모든 명령은 저장소 루트(`~/Autonomous-Driving-2026`) 기준입니다.

---

## 0. 오늘 밤 PC에서 준비되는 것 (Claude)

| 항목 | 상태 (11/25 저녁) | 아침까지 |
|---|---|---|
| 맵 툴 (`map_offline.py` 등) | 짧은 코스에서도 되도록 고침. 정답 비교 벽 오차는 연습 코스 3.3 cm, 50–90 m 코스 5.6–7.6 cm (`MAP_TOOL.md` 1절) | 완료 |
| 위치 추정 (`utils/localizer.py`) | 11/24 실제 기록에서 62 s 내내 유효, 오차 중앙값 2.9 cm. 시뮬레이터 4–5 m/s 정답 비교 2.8–10.6 cm. 스캔당 2–3 ms(PC) | 완료 |
| 차 쪽 연결 (`agent_node.py`, `agent_preview.yaml`) | 첫 스캔에서 코스를 따라 시작 위치를 찾고, 반대 방향이면 `_rev` 코스로 자동 전환. 위치를 잃으면 LiDAR만으로 주행. `/loc_pose` 기록 | PC 재생 검사(`tools/agent_replay.py`) 뒤 완료 |
| IL 13 (위치 정보 입력, 짧은 코스 50 %) | 16:06 시작, 약 4시간 | 결과를 보고 Claude가 PPO 14 설정 |
| PPO 14 (IL 13에서 시작) | IL 13 뒤 밤새 | 아침에 가장 좋은 모델을 `agent_preview.yaml`에 넣고 push |
| LiDAR 전용 모델 | PPO 12 u190(`12190`, 안정) / u272(`12272`, 약 12 % 빠름). `agent_ilppo.yaml` / `agent_ilppo_fast.yaml` | 완료 |

아침에 `git pull` 하면 모든 것이 들어옵니다. 계획 B 모델이 준비되지 않으면 4번은 건너뛰고 1–3번만 합니다.

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
2. agent 로그를 봅니다.
   - `[preview] localization: start at x .. y .. yaw ..`가 나와야 합니다(시작 위치 찾기).
   - `[preview] valid ...% ... step X ms`가 5초마다 나와야 합니다.
3. 차를 손으로 밀어 코스를 한 바퀴 돕니다. `valid`가 100 % 근처를 유지해야 합니다.
4. **`step`이 10 ms를 넘으면** `agent_preview.yaml`에서 `loc_stride: 4`로 바꿉니다(Jetson이 PC보다 느림).

---

## 5. 위치 정보 모델 주행 (계획 B, 30분) — 4번이 통과했을 때만

```bash
MODEL=preview tools/drive_session.sh      # agent_preview.yaml: 아침에 Claude가 넣은 모델 (PPO 14 또는 IL 13)
```
- `speed_limit` 2.5에서 시작해 3 → 4 → 5로 올립니다. 매번 3바퀴 이상 돌고 `--finish`로 기록합니다.
- **위치를 0.3 s 넘게 잃으면 차가 스스로 멈춥니다**(속도 0, 조향은 모델 그대로). 위치를 다시 찾으면 이어서 달립니다. 로그에는 `[preview] localization lost: stopping after 0.3 s`가 나옵니다.
  - 이유: 11/25 점검에서 IL 13은 위치 없이 처음부터 끝까지 달리면 완주가 일반 트랙 9 %, 짧은 코스 53 %였습니다(위치가 있으면 93 %, 97 %). 학습 때 끊김을 0.5–3 s로 짧게만 넣었기 때문입니다.
  - PPO 14부터는 긴 끊김과 위치 없는 에피소드를 넣어 학습합니다(`configs/task/sim_override_preview_lost.yaml`).
  - `lost_stop: 0`으로 바꾸면 멈추지 않고 LiDAR만으로 계속 달립니다. PPO 14의 위치 없는 주행 성능을 보고 정합니다.
- 로그에 `[preview] car faces the other way: using ..._rev.npz`가 나오면 코스 방향을 바꾼 것입니다. 실제 주행 방향과 맞는지 확인합니다.
- 주행 평가(`eval_drive.py`)에 `localization while driving: valid ...%`가 나옵니다. 위치 추정의 실제 성적입니다.
- 같은 상한에서 3번 LiDAR 전용 모델과 랩 타임, 벽과의 거리를 비교합니다(`eval.txt`).

---

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
| 4 위치 추정 | 20분 |
| 5 계획 B | 30분 |
| 합계 | 약 2시간 10분 (배터리 교체 포함 2.5시간) |
