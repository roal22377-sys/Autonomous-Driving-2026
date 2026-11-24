# 학습·평가·분석 도구 사용법

다른 PC에서도 그대로 따라 할 수 있게 정리한 문서입니다(2026-11-23 기준).
학습 절차와 설정의 이유는 [`TRAINING_S2R.md`](TRAINING_S2R.md)를 보세요.
IL 방법 정리는 [`IL_METHODS_1124.md`](IL_METHODS_1124.md), PPO 보상과 계획은 [`PPO_PLAN_1124.md`](PPO_PLAN_1124.md)에 있습니다.

- 모든 명령은 **`src/f1tenth` 폴더에서** 실행합니다(설정 파일과 `results/`를 상대 경로로 찾습니다).
- 앞에 `OMP_NUM_THREADS=1`을 붙입니다. 프로세스 여러 개가 각자 스레드를 늘려 CPU를 서로 뺏는 것을 막습니다.
- 아래 예시의 `PY`는 학습 환경의 파이썬입니다: `PY=~/venv/f1tenth/bin/python`
- GPU는 필요 없습니다. 전부 CPU로 돕니다.

---

## 1. 새 PC 준비 (한 번만)

```bash
# 1) 코드 받기
git clone https://github.com/roal22377-sys/Autonomous-Driving-2026.git ~/Autonomous-Driving-2026
cd ~/Autonomous-Driving-2026                 # 이미 있으면: git pull

# 2) 파이썬 환경 (Python 3.8 필요, 약 5분) -> ~/venv/f1tenth
tools/setup_train_env.sh

# 3) 트랙 만들기 (git에 없음, PC마다 한 번. 같은 코드면 어느 PC에서나 같은 트랙이 나옴)
cd src/f1tenth
PY=~/venv/f1tenth/bin/python
OMP_NUM_THREADS=1 $PY -m utils.track_gen 16 20                # 트랙 782개 (학습 736 + 시험용 w105 46), 약 5분
OMP_NUM_THREADS=1 $PY -u eval_open_space.py --make --workers 20   # 막다른 공간 시험 트랙 132개, 약 1분
$PY -m utils.track_gen --check                                 # "322 up to date, 0 stale"이면 준비 끝
```

**workers 수 정하기 (메모리 기준).** 시뮬레이터 프로세스 하나가 약 0.4 GB입니다.

| PC 메모리 (WSL이면 `free -g`의 total) | IL `--workers` | PPO `--workers` + `--eval_workers` | 분석 도구 `--workers` |
|---|---|---|---|
| 16 GB | 20 (기본값) | 16 + 8 (기본값) | 8 |
| 32 GB | 28 | 24 + 12 | 16 |
| 8 GB | 8 (`--max_data 2e6`도 함께) | 8 + 4 | 4 |

IL은 학습 데이터(400만 샘플, 약 2.2 GB)를 추가로 씁니다. CPU 스레드 수보다 workers를 많이 잡을 필요는 없습니다.

**코드가 바뀌어 트랙 생성기가 달라지면** 같은 `track_gen` 명령을 다시 실행하세요. 바뀐 트랙만 새로 만들어집니다.
옛 트랙이 남아 있으면 IL/PPO 학습이 시작을 거부하고 이 명령을 안내합니다.

---

## 2. IL 학습 — `train_il_s2r.py`

```bash
OMP_NUM_THREADS=1 nohup $PY -u train_il_s2r.py --algo_idx 11 \
    --task_config configs/task/f1tenth_lidar88.yaml --hidden_dim 512 \
    --expert_steps 1.5e6 --dagger_steps 8e5 --dart_std 0.08 --grad_steps 20000 \
    > results/imitation_11_train.log 2>&1 &
tail -f results/imitation_9_train.log           # 진행 보기 (Ctrl+C는 보기만 멈춤, 학습은 계속)
```

- 진행 순서: 라운드 0(expert 주행 데이터) → 라운드 1, 2, …(DAgger: 정책이 운전하고 expert가 정답을 붙임).
  라운드마다 행동 복제 학습 → 평가 → 최고 점수 라운드 저장.
- **점수** = (일반 트랙 2랩 완주율 + 큰 막다른 공간 트랙 2랩 완주율) ÷ 2.
- **자동 종료**: 라운드 3 이후 점수 +1%p(같은 점수면 짝지은 랩 타임 1% 단축)가 2라운드 연속 없으면 멈춤. 최대 20라운드.
- **중간에 멈췄으면** 같은 명령에 `--resume`을 붙입니다(`results/imitation/<번호>/data/`의 라운드 데이터 사용).
- 결과: `results/imitation/<번호>/sim2real/`(차가 쓰는 모델), `checkpoint/model_round*.pt`(라운드별), `train_log.pkl`.
- PC에 맞출 옵션: `--workers`, `--bc_threads`(행동 복제 단계 스레드, 기본 16), `--max_data`(메모리).
- 로그에서 볼 줄: `[round N] ... EVAL ... 2-lap success X% ... open-space ... score Y`, `noticeable improvement` 여부, `[stop]`.

## 3. PPO 학습 — `train_ppo_s2r.py`

```bash
OMP_NUM_THREADS=1 nohup $PY -u train_ppo_s2r.py --il imitation/11 --algo_idx 2 --force_start \
    > results/ppo_2_train.log 2>&1 &
```

- **시작 조건**: IL이 일반 트랙 98% 이상, 막다른 공간 95% 이상이어야 시작합니다. 아니면 이유를 출력하고 멈춥니다.
  IL이 더 개선되지 않는 상태라면 `--force_start`로 시작합니다(11/23 결정).
- **중간에 멈췄으면** 같은 명령에 `--resume`을 붙입니다(5 업데이트마다, 그리고 평가 직후 `checkpoint/resume.pt` 저장).
- **예전 설정과 비교**: `--baseline`.
- **시뮬 조건만 바꾸기**(IL은 다시 학습하지 않음): `--sim_override 파일.yaml`. 예:
  ```yaml
  # 중간 조향 원선회에서 좌우가 같았을 때
  sim2real:
    steer_scale_left: [0.9, 1.05]
    steer_scale_left_nominal: 1.0
  ```
  바꿀 수 있는 것은 `sim2real` 전체와 `speed_limit`의 `range`/`uncapped_prob`/`nominal`뿐입니다.
  LiDAR, 조향 기준값처럼 정책 입력·출력에 관계된 값은 거부됩니다.
  접지 μ는 `configs/task/dynamic.yaml`의 `mu`입니다(IL 학습이 돌고 있을 때는 고치지 마세요).
- 결과: `results/ppo_s2r/<번호>/sim2real/`(IL보다 좋을 때만 저장), `checkpoint/`, `train_log.pkl`, `backup/ppo_args.yaml`(사용한 설정).

자주 바꾸는 옵션 (기본값 / 의미):

| 옵션 | 기본 | 의미 |
|---|---|---|
| `--total_steps` | 12e6 | 총 시뮬 스텝 |
| `--eval_every` | 10 | 몇 업데이트마다 평가 (평가가 오래 걸리면 늘림) |
| `--lr` / `--lr_end` | 5e-5 / 1e-5 | actor 학습률 (선형 감소) |
| `--clip` / `--max_kl` | 0.15 / 0.015 | 한 번에 바뀌는 양 |
| `--log_std_steer` / `--log_std_speed` | −2.3 / −1.5 | 탐색 노이즈 (std 0.10 / 0.22), `--log_std_decay` 0.5만큼 줄어듦 |
| `--bc_start` / `--bc_end` | 0.5 / 0.02 | IL 쪽으로 당기는 힘 |
| `--crash_adapt` / `--crash_up` / `--crash_down` / `--crash_max` | 1 / 1.5 / 0.8 / 2000 | 평가 완주율이 IL보다 2 %p 넘게 낮으면 충돌 벌점 ×1.5, IL 이상이면 ×0.8 (500 아래로는 안 내림) |
| `--crash_target` | il | 충돌 벌점의 목표 완주율. `il` = IL 완주율 고정, `best` = 래칫(저장된 최고 모델의 완주율, 올라가기만 함) |
| `--crash_start` | 없음 | 시작할 때 충돌 벌점 (없으면 `--collision_cost`, 하한은 계속 `--collision_cost`) |
| `--init` | 없음 | PPO 체크포인트에서 시작 (예: `ppo_s2r/2/checkpoint/model_1760000.pt`). 저장 기준은 계속 `--il`과 비교 |
| `--baseline_from` | 없음 | 같은 `--il`로 돌린 이전 PPO의 IL 기준 평가 재사용 (약 8분 절약) |
| `--bc_adapt` | 0 | 1이면 완주율 하락 때 IL 쪽 당김을 2배씩 (PPO 2 방식) |
| `--bc_max` | 8 | 완주율이 IL보다 떨어질 때 당기는 힘을 2배씩 올리는 상한 (bc_start의 8배) |
| `--collision_cost` | 500 | 충돌 벌점 (PPO 1의 200은 속도를 위해 완주율을 내줌) |
| `--bc_speed` | 0.25 | 속도 출력은 조향의 0.25배만 IL 쪽으로 당김 (1 = 조향과 같음) |
| `--track_focus` | 3.0 | 학습 트랙 뽑는 가중치 = 1 + 3 × 그 트랙의 최근 실패율 (0 = 균등) |
| `--full_eval` | 0 | 0: 최악 조건·막다른 공간 평가는 저장 후보(일반 평가에서 최고 이상)일 때만 실행해 평가 시간 약 절반. 1: 매번 실행(곡선이 다 그려짐) |
| `--stuck_time` / `--stuck_dist` / `--backward_dist` | 3.0 / 0.5 / 2.0 | 3초간 0.5 m도 못 나아가거나 2 m 역주행하면 충돌로 끝냄 (0 = 끔, 평가는 항상 원래 규칙) |
| `--time_cost` | 0.1 | 스텝당 시간 벌점 (= 랩 타임 최소화) |
| `--steer_rate_cost` | 3.0 | 조향 급변(흔들림) 벌점 |
| `--wall_cost` / `--wall_margin` | 5.0 / 0.30 | 벽 여유 벌점 |
| `--lat_cost` / `--lat_grip_share` | 2.0 / 0.8 | 접지 한계(0.8 μg) 넘는 횡가속 벌점 |
| `--lap_bonus` | 0 | 랩 완주 보너스 (속도가 안 오를 때 2로 시험) |
| `--uncapped_prob` | 0.5 | 속도 상한 없는 에피소드 비율 |

## 4. 분석 도구 (평가만 함, 학습 없음)

### 4-1. 학습 곡선 — `plot_training.py`

```bash
$PY plot_training.py imitation/11            # -> results/imitation/11/training.png + 라운드별 요약 출력
$PY plot_training.py ppo_s2r/1 ppo_s2r/2    # 여러 개 (각자 그림)
```

- IL: 라운드별 일반/막다른 공간 완주율과 점수(98%/95% 기준선 표시), 랩 타임, 검증 오차, 데이터 양. 저장된 라운드는 초록색.
- PPO: critic 설명력(0.6 기준선, 먼저 학습 구간 표시), KL·clip 비율, 탐색 노이즈와 BC 가중치, 학습 에피소드 충돌률·완주율·평균 속도,
  보상 항목별 크기, 평가(일반/최악 조건/막다른 공간 완주율, 최고 모델 대비 단축 시간, 저장 시점).
- 학습 중에도 실행할 수 있습니다(라운드/업데이트마다 로그가 갱신됨).

### 4-2. 실패 장면 — `plot_failures.py`

```bash
$PY plot_failures.py imitation/11                         # IL 평가 트랙 x 평가 조건 3개 (132회)
$PY plot_failures.py imitation/11 --tracks open:big       # 큰 막다른 공간 시험 트랙 (상한 8, 5)
$PY plot_failures.py imitation/11 --tracks 'installation/maps_gen/Sochi*.npz' --caps 5
$PY plot_failures.py ppo_s2r/1 --all --max_plots 10      # 완주한 주행도 그림
```

- 출력: `results/<모델>/failures/<트랙>@<상한>_<출발>.png`, 전체 요약 `summary.png`, 실패 표(어디서, 속도, 조향, 충돌 위치).
- 그림 한 장: 트랙 전체 궤적(속도 색), 충돌 직전 4초 확대(조향 색, expert 레이스 라인, 주황 점선 = 막다른 공간),
  마지막 20초의 속도·조향·중앙선 거리.
- 충돌 위치 분류: `run-off`(막다른 공간 안), `pocket`(옆 공간), `corridor`(정상 트랙 폭 안에서 벽).
- 옵션: `--workers`(기본 8), `--max_plots`(기본 40), `--start center|random`(`--caps`와 함께).

### 4-3. 모델 비교 보고서 — `compare_models.py`

```bash
$PY -u compare_models.py imitation/11 ppo_s2r/1             # 첫 번째가 기준
$PY -u compare_models.py expert imitation/11 ppo_s2r/1      # expert도 함께 (상한선 확인)
$PY -u compare_models.py expert imitation/11 --quick --max_tracks 6   # 1랩, 몇 개만 (빠른 확인, 1분 이내)
```

- 항목: 일반 트랙(평가 조건 3개), 최악 조건(지연 0.2 s, μ 0.4 등, 상한 8), 막다른 공간 트랙(상한 8, 5). 모두 2랩.
- **짝지은 랩 타임**: 기준 모델과 둘 다 완주한 주행끼리만 비교해서 "x% 빠름/느림"으로 표시합니다.
- 출력: 화면 표 + `results/reports/compare_<날짜>_<모델들>.md`(트랙별 랩 타임, 실패 목록 포함).
- 모델마다 자기 설정(LiDAR 칸 수 등)으로 돌기 때문에 run 8(54칸)과 run 9(88칸)도 같이 비교할 수 있습니다.
- 전체 비교는 모델당 약 350회 주행이라 workers 8 기준 수십 분 걸립니다.

### 4-4. 막다른 공간 점검 — `eval_open_space.py`

```bash
$PY -u eval_open_space.py expert imitation/11                       # 대조군 none / train / big, 상한 5·8, 2랩
$PY -u eval_open_space.py imitation/11 --levels none big --caps 5
```

- 같은 트랙에서 막다른 공간만 다른 세 묶음: `none`(없음, 대조군), `train`(학습 크기 2~7 m), `big`(큰 것 4~7 m만).
  `none` 대비 `big`의 충돌 증가가 "넓은 막다른 공간 때문에 생긴 실패"입니다.
- 표: 완주율, 충돌 위치별 수(run-off / pocket / corridor), 막다른 공간 진입 횟수, 충돌 직전 속도.

### 4-5. 기본 평가 — `evaluate_s2r.py`

```bash
$PY -u evaluate_s2r.py imitation/11 ppo_s2r/1 --workers 8
```

일반 평가 트랙에서 실차 조건(`eval`)과 원래 시뮬레이터(`ideal`) 완주율과 랩 타임을 봅니다. 위 비교 보고서가 더 자세합니다.

---

## 5. 모델 보관과 실차 사용

### 학습이 끝난 모델 보관 (git)

`results/`는 git에서 제외돼 있어서, 차에서 쓸 파일만 골라 강제로 올립니다(데이터·체크포인트는 크므로 제외).

```bash
cd ~/Autonomous-Driving-2026
git add -f src/f1tenth/results/imitation/11/sim2real src/f1tenth/results/imitation/11/backup src/f1tenth/results/imitation/11/README.md
git commit -m "IL run 11 model" && git tag model-il11 && git push && git push --tags
# PPO도 같은 방식: results/ppo_s2r/<번호>/{sim2real,backup}, 태그 model-il11-ppo<번호>
```

나중에 특정 버전으로 되돌리기: `git checkout model-il11 -- src/f1tenth/results/imitation/11`

### 차에서 모델 고르기

| 실행 | 쓰는 설정 | 모델 |
|---|---|---|
| `tools/drive_session.sh` | `config/agent.yaml` | 지금까지 쓰던 것 (run 8) |
| `MODEL=il tools/drive_session.sh` | `config/agent_il.yaml` | IL만 (run 11) |
| `MODEL=ilppo tools/drive_session.sh` | `config/agent_ilppo.yaml` | IL + PPO (`algo_idx`를 저장된 PPO 번호로 맞출 것) |

`tools/run_car.sh`도 같습니다(`MODEL=il tools/run_car.sh use_agent:=true`).
새 모델 설정은 `steer_gain` 1.0, `speed_min` 0, `speed_limit` 3.0으로 시작합니다. 잘 돌면 5.0 → 대회 8.0으로 올립니다.
주행 뒤 `tools/drive_session.sh --finish`로 로그를 가져오면 `tools/eval_drive.py` 결과가 나옵니다.

---

## 6. 자주 나는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `It could not load track ...` | `yamldataclassconfig`가 2.x. `pip install yamldataclassconfig==1.5.0` (설치 스크립트는 1.5.0으로 고정) |
| `No module named 'wandb'` | `main_il.py`가 import함. `pip install wandb` |
| `... tracks were made by an older track generator` | 트랙 재생성: `$PY -m utils.track_gen 16 <workers>` |
| `no open-space test tracks` | `$PY -u eval_open_space.py --make` |
| `IL policy ... is not solid enough for PPO` | IL이 98%/95% 미만. IL을 더 돌리거나, 더 개선되지 않으면 `--force_start` |
| 학습이 갑자기 죽음 (`Killed`), 스왑 사용 급증 | 메모리 부족. `--workers`(PPO는 `--eval_workers`도), IL은 `--max_data`를 낮춤 |
| `Can't get attribute ... <stdin>` 같은 오류 | 멀티프로세스 도구를 `python - <<EOF`로 실행하면 생김. 파일로 저장해서 실행 |
| 여러 줄 `torch.load ... weights_only` 경고 | 무해함 (torch 2.4 안내 메시지) |

## 7. 파일 위치 요약

| 위치 | 내용 |
|---|---|
| `configs/task/f1tenth.yaml` | 기본 설정 (LiDAR 54칸) |
| `configs/task/f1tenth_lidar88.yaml` | run 9~11 설정 (LiDAR 88칸). LiDAR 외에는 기본 설정과 같게 유지 |
| `configs/task/dynamic.yaml` | 차량 물리 (접지 μ 등) |
| `installation/maps_gen/` | 학습 트랙 (생성, git 제외) |
| `installation/maps_open_test/` | 막다른 공간 시험 트랙 (생성, git 제외) |
| `results/imitation/<번호>/`, `results/ppo_s2r/<번호>/` | 학습 결과 |
| `results/reports/` | 비교 보고서 |
| `requirements_train.txt`, `../../tools/setup_train_env.sh` | 학습 환경 버전과 설치 |

### 실차 로그 지도화 (사고 지점 보기)
```bash
python3 tools/log_map.py logs/<run>_extracted out.png <시작 s> <끝 s> [표시할 시각 s ...]   # LiDAR 스캔 매칭(ICP)으로 구간 지도 + 궤적 + 조향
python3 tools/log_view.py logs/<run>_extracted <출력 폴더>                               # 자율 구간마다 사고 0.5~2 s 전 LiDAR 장면
```
odom만으로 그리면 방향 오차로 지도가 번지므로 `log_map.py`(스캔 매칭)를 씁니다. 차가 제자리에서 돌거나 끼어 있으면 매칭이 실패할 수 있습니다.
