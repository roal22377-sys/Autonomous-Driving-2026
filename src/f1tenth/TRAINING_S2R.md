# sim2real 학습 파이프라인 (IL → PPO)

> 새 PC 준비, 각 도구의 명령·옵션·출력, 모델 보관과 실차 선택, 자주 나는 문제는 [`TOOLS.md`](TOOLS.md)에 정리했다.
> IL run 11에 적용한 방법·제한·근거와 시험했지만 넣지 않은 것은 [`IL_METHODS_1124.md`](IL_METHODS_1124.md)에 정리했다.
> PPO 보상 설계, 안정성 장치, 진행 순서, 실차 시험과 조정 기준은 [`PPO_PLAN_1124.md`](PPO_PLAN_1124.md)에 정리했다.

> 2026-11-19 실차 점검(조향 지연·왼쪽 조향 잘림·IL run 6 실차 평가)과 PPO 보상·하이퍼파라미터 논의는
> [`NOTES_1119_sim2real_rl.md`](NOTES_1119_sim2real_rl.md)에 정리했다. 학습 전에 먼저 읽을 것.

실제 주행 로그(`logs/run_1117_14*`)에서 찾은 문제를 시뮬레이터에 반영해서 학습한다.
- 조향 지연 약 0.2 s 때문에 좌우로 흔들림(weaving)
- 학습 트랙(폭 2.2 m 고정)에 없던 넓은 구간·열린 옆 공간
- 실제 가속 성능 약 3 m/s²

## 바뀐 것

| 항목 | 내용 | 파일 |
|---|---|---|
| 조향/속도 지연 | 학습: 0.10~0.25 s / 0~0.05 s 무작위, 평가: 0.2 s / 0.03 s | `utils/env.py`, `configs/task/f1tenth.yaml` `sim2real` |
| 차 성능 | a_max 2.5~3.5 m/s² (실측 2.7), 마찰 ×0.85~1.1, 차 크기 0.30×0.60 m | `configs/task/dynamic.yaml` |
| 서보 모델 | bang-bang(평균 35% 덜 꺾임)을 위치 제어 + 속도 제한으로 수정 | `installation/.../dynamic_models.py` `pid_steer` |
| 센서 노이즈 | 실측 기반: 라이다 0.008 m, 무효값 0.3%, 튀는 값, 마운트 ±1°, odom 0.06 m/s, 에피소드마다 ×0.5~2 | `utils/lidar.py`, `f1tenth.yaml` |
| 트랙 | 23개 맵 × 정/역방향 × (원래 폭 + 무작위 폭 6개) = 322개. 폭 1.7~4.4 m, 열린 옆 공간 | `utils/track_gen.py` (`python -m utils.track_gen 6 22`) |
| 레이싱 라인 | 최소 곡률 → 랩 타임 직접 최소화(late apex 등). 벽과 0.55 m 여유 | `utils/raceline.py` |
| expert | 레이싱 라인 추종 + 조향 지연 보정(보내놓은 명령으로 위치 예측) + 라인 이탈 시 감속 합류 | `utils/rule_based.py` `RacelinePlanner` |
| 관측 (76) | 라이다 54 + 속도 1 + **최근 명령 10개(20)** + **속도 상한 1** | `utils/observation.py`, `agent_node.py` |
| 속도 상한 | 학습 때 에피소드마다 2~8 m/s 무작위 (25%는 상한 없음). 실차 `speed_limit`을 정책이 앎 | `f1tenth.yaml` `speed_limit` |
| 에피소드 | 2바퀴 완주 = 성공, 스텝 수는 트랙 길이로 자동 | `f110_env.py`, `env.py` |
| IL | DART(0.08) → DAgger 3회, 조향 가중 MSE, cosine lr, 평가 트랙 분리(`_w105`) | `train_il_s2r.py` |
| PPO | IL 초기화, 비대칭 critic(시뮬레이터 정답 상태 15개), critic 먼저 학습(EV ≥ 0.6까지), IL 쪽 BC 항 감소(완주율 떨어지면 다시 올림), return 정규화, 진행거리 − 시간 벌점(= 랩 타임 최소화) + 충돌 −200 + 조향 급변·벽 여유·접지 한계 벌점, 조향/속도 노이즈 분리, 짝지은 랩 타임 + 최악 조건 완주율 + 열린 공간 시험 트랙 2랩 완주율(IL보다 낮으면 저장 안 함)로 모델 선택 | `train_ppo_s2r.py` |
| 트랙 버전 | 트랙 파일마다 생성기 서명 저장. 생성기가 바뀌면 다시 실행할 때 옛 파일만 새로 만들고, IL/PPO는 옛 트랙이면 시작을 거부 | `utils/track_gen.py` |
| 코너 run-off 크기 | 11/23 실차(넓은 막다른 공간으로 직진해 벽 충돌) 대응. 길이 2~5 m → **2~7 m**, 트랙보다 넓은 폭 0~1 m → **0~2.5 m** (run 8이 4~7 m 공간에서 27개 중 13개 트랙 충돌) | `utils/track_gen.py` |
| 열린 공간 점검 | 모든 코너에 막다른 공간(run-off)이 있는 시험 트랙(같은 트랙의 run-off 없는 대조군 포함)에서 2랩, 그 안으로 들어가 박는지 집계 | `eval_open_space.py` |

## 실행: 11/23~24 보정 반영 IL run 11 → PPO (src/f1tenth에서)

11/23에 보정값(`max_steer` 0.41, 좌우 게인, 지연 0.05~0.20 s, μ 0.5), 정책 속도 범위(`min_speed` 0.5),
expert, 트랙 생성기(코너 run-off, 사각 alcove, 벽 요철)가 바뀌었다. run 8 모델과 데이터는 행동의 의미가 달라
쓸 수 없으므로 `--init` 없이 처음부터 학습한다.

run 9부터 라이다 88칸(`configs/task/f1tenth_lidar88.yaml`, 관측 114): 정면 ±20° 1.25°, 20~60° 2.5°, 60~100° 5°,
100~135° 8.75°. 8 m 앞에서 칸 폭이 0.70 m(54칸) → 0.17 m라 멀리 있는 코너, 막다른 공간의 끝 벽과 코너 출구를
더 정확히 구분한다. 최대 거리는 10 m 그대로. 54칸(`f1tenth.yaml`)과 60칸(`f1tenth_lidar60.yaml`)은 비교용으로 남긴다.
학습 PC(i9-14900KF, WSL 메모리 15.5 GB + 스왑 4 GB)에 맞춤: `--max_data` 4e6(8e6은 약 23 GB 필요),
workers 20(프로세스당 약 0.4 GB, 20개 동시 실행 최고 9.1 GB + 데이터 2.2 GB), BC 스레드 16, 프로세스당 트랙 캐시 4개.

expert(11/23): 명령한 조향으로 도는 데 필요한 횡가속이 접지의 70%를 넘지 않게 감속하는 규칙 추가
(출발 직후 합류 중 7 m/s로 가속하다 미끄러지던 문제). expert만 2랩: 평가 트랙 94.7% → 100%,
큰 막다른 공간 트랙 87.5% → 97.7%. PPO는 IL이 평가 트랙 98%·막다른 공간 95% 이상일 때만 시작(`--force_start`로 무시).

```bash
# 0. 트랙: 생성기가 바뀐 파일만 다시 만든다 (전체 약 1~3분). 상태만 보려면 --check
OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -m utils.track_gen 6 20
OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -u eval_open_space.py --make --workers 20   # 열린 공간 시험 트랙

# 1. IL run 11 (라이다 88칸)
#    라운드마다: 일반 평가(상한 8 중앙/8 랜덤/5 랜덤 출발, 2랩) + 열린 공간 big 트랙(상한 8·5, 2랩)
#    점수 = 두 완주율의 평균. 최고 점수 라운드를 저장(같으면 짝지은 랩 타임이 빠른 쪽).
#    조기 종료: 라운드 3 이후 점수 +1%p 또는(같은 점수에서) 랩 타임 1% 단축이 2라운드 연속 없으면 멈춤, 최대 20라운드
OMP_NUM_THREADS=1 nohup ~/venv/f1tenth/bin/python -u train_il_s2r.py --algo_idx 11 --workers 20 \
    --task_config configs/task/f1tenth_lidar88.yaml --hidden_dim 512 \
    --expert_steps 1.5e6 --dagger_steps 8e5 --dart_std 0.08 --grad_steps 20000 \
    > results/imitation_11_train.log 2>&1 &

# 2. 평가: 일반 트랙 + 열린 공간 시험 트랙 (expert와 비교)
OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -u evaluate_s2r.py imitation/11 --workers 20
OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -u eval_open_space.py expert imitation/11 --workers 20

# 3. PPO (2번의 eval_open_space.py --make 필요: 모델 선택에 big 시험 트랙 2랩 완주율을 씀)
#    기준선(11/23 이전 설정) 1회 → 추천 설정. 이후 옵션을 하나씩 바꿔 비교 (seed 2~3개)
#    PPO는 IL 모델의 backup/f1tenth.yaml(88칸)을 그대로 쓴다
OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -u train_ppo_s2r.py --il imitation/11 --algo_idx 0 --baseline
OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -u train_ppo_s2r.py --il imitation/11 --algo_idx 1
#    멈췄으면 같은 명령에 --resume (5 업데이트마다 checkpoint/resume.pt 저장)
#    하나씩 끄고 비교: --ret_norm 0 / --time_cost 0 / --log_std_steer -2 --log_std_speed -2 --uncapped_prob 0.25
#                      / --wall_cost 0 --lat_cost 0,  랩 보너스는 --lap_bonus 2 로 켬
OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -u evaluate_s2r.py imitation/11 ppo_s2r/0 ppo_s2r/1 --workers 20
OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -u eval_open_space.py expert imitation/11 ppo_s2r/1 --workers 20
```

- PPO 자원(이 PC): 롤아웃 프로세스 `--workers 16` + 평가 프로세스 `--eval_workers 8`. 롤아웃 프로세스는 평가 중에도
  떠 있으므로 둘의 합으로 메모리를 잡는다(20+20은 15.5 GB를 넘음).
- 보상 크기(expert 2랩에 기본 계수 적용, 11/23): 스텝당 진행 +1.37, 시간 −0.10(7.3%), 조향 급변·벽 여유·접지 한계는
  각각 진행의 0.3% 이하. 잘 달리면 벌점이 거의 붙지 않고, 흔들리거나 무리할 때만 커진다.
- PPO 로그(업데이트마다): explained variance, clip 비율, KL, 조향/속도 std, return 스케일, 충돌률, 평균 속도,
  보상 항목별 스텝당 값. `results/ppo_s2r/<idx>/train_log.pkl`, 사용한 설정은 `backup/ppo_args.yaml`.
- 새 모델을 차에 올릴 때 `agent.yaml`의 `steer_gain_left/right`는 1.0부터 (1.35/1.12는 run 8용 보정).

## 분석 도구 (src/f1tenth에서, 학습 없이 평가만)

| 도구 | 용도 | 예 |
|---|---|---|
| `plot_training.py` | IL 라운드별 / PPO 업데이트별 학습 곡선 그림 + 요약 | `python plot_training.py imitation/11` → `results/imitation/11/training.png` |
| `plot_failures.py` | 실패한 주행을 트랙 위 궤적(속도, 조향, run-off)과 시간 그래프로, 충돌 위치 분류 | `python plot_failures.py imitation/11 --tracks open:big` → `results/imitation/11/failures/` |
| `compare_models.py` | 모델 비교 보고서: 일반 / 최악 조건 / 열린 공간 2랩 완주율, 짝지은 랩 타임, 트랙별 랩 타임 | `python -u compare_models.py imitation/11 ppo_s2r/1` → `results/reports/` |
| `eval_open_space.py` | 막다른 공간 점검 (대조군 none / train / big) | `python -u eval_open_space.py expert imitation/11` |
| `train_ppo_s2r.py --sim_override x.yaml` | IL을 다시 학습하지 않고 PPO 시뮬 조건만 변경 (`sim2real`, `speed_limit`의 range/uncapped_prob/nominal) | 중간 조향 원선회 결과 반영 등. 접지 μ는 `configs/task/dynamic.yaml`(IL 학습이 끝난 뒤 수정) |

## 실차: IL만 / IL+PPO 버전 나눠 쓰기

- `config/agent_il.yaml` = IL run 9, `config/agent_ilppo.yaml` = IL+PPO(algo_idx를 저장된 PPO 번호로). 기존 `agent.yaml`(run 8)은 그대로.
- 둘 다 `steer_gain` 1.0, `speed_min` 0, `speed_limit` 3.0부터.
- 선택: `MODEL=il tools/drive_session.sh` / `MODEL=ilppo tools/drive_session.sh` (`tools/run_car.sh`도 같음). MODEL 없으면 agent.yaml.
- 학습이 끝난 모델은 `results/<model>/{sim2real,backup,README.md}`만 커밋(데이터·체크포인트 제외)하고 태그를 단다:
  IL만 `model-il11`, IL+PPO `model-il11-ppo<번호>` → 나중에 `git checkout model-il11 -- src/f1tenth/results/imitation/11`로 언제든 복원.

## 예전 실행 기록 (run 8, 11/21)

```bash
# 0. 트랙 생성 (한 번, 약 3분)
OMP_NUM_THREADS=1 ~/venv/torch210/bin/python3 -m utils.track_gen 6 22

# 1. IL (run 8 = 다음 학습 "A 방법": 512x2 신경망 + CAPS 떨림 억제, 라운드별 데이터 저장)
#    run 8부터 11/19 실차 점검 반영 (NOTES_1119): 조향 지연 범위 0.05~0.30 s, 행동 이력 12 (obs_dim 80),
#    왼쪽 조향 한계 0.428~0.482 rad 랜덤 (eval 0.428 = 지금 차, agent_node와 같은 잘림), 비대칭 critic 상태 15개
OMP_NUM_THREADS=1 nohup ~/venv/torch210/bin/python3 -u train_il_s2r.py --algo_idx 8 --workers 22 \
    --hidden_dim 512 --caps_t 0.3 --caps_s 0.3 \
    --expert_steps 1.5e6 --dagger_iters 1000 --dagger_steps 8e5 --dart_std 0.08 --grad_steps 20000 \
    > results/imitation_8_train.log 2>&1 &
#    멈췄다가 이어서: 같은 명령에 --resume (results/imitation/8/data/round*.npz 사용)

# 2. PPO 미세조정
OMP_NUM_THREADS=1 ~/venv/torch210/bin/python3 -u train_ppo_s2r.py --il imitation/8 --algo_idx 1 --workers 22

# 3. 비교 (같은 트랙·출발 위치, 실차 조건 eval / 원래 시뮬레이터 ideal, 속도 상한 8·3)
OMP_NUM_THREADS=1 ~/venv/torch210/bin/python3 -u evaluate_s2r.py imitation/6 imitation/7 imitation/8 ppo_s2r/1
```

기준 (IL run 6, 실차 조건): 상한 없음 0%, 3 m/s 33% 완주 (트랙 3개 샘플). 원래 시뮬레이터에서는 100%.

## 결과 기록

### IL run 9 (11/23, 88칸, 감속 규칙 'cap') — 라운드 5에서 조기 종료, 최고 라운드 3
| 라운드 | 일반 2랩 | 막다른 공간 2랩 | 점수 | 조향 떨림 | 2랩 시간 |
|---|---|---|---|---|---|
| 0 | 10.6% | 8.0% | 9.3 | 0.005 | 104 s |
| 3 (저장) | 73.5% | 43.2% | 58.3 | 0.029 | 218 s |
| 5 | 72.0% | 33.0% | 52.5 | 0.028 | 199 s |

원인 분석(데이터·expert 평가만): expert의 "꺾는 만큼 감속" 규칙이 정책 주행 상태 정답의 34 %를 감속으로 만들어
느려짐(정답 속도 4.0 → 2.9 m/s). 정책은 큰 조향이 필요한 곳에서 필요한 조향의 35~60 %만 꺾고(주 실패 원인),
직진 근처에서는 잡음처럼 떨림(방향 일치 64 %). 학습/검증 오차가 같이 높음 = 정답을 관측으로 맞추기 어려움
(조향 지연·μ 숨은 정보는 그중 13 %; 레이스 라인 위치를 정책이 볼 수 없는 것이 큰 부분으로 추정).
CAPS·과거 명령 잡음·신경망 폭 1024는 짧은 BC 확인에서 효과 없음.

### IL run 10 (11/23~24) — 라운드 4에서 중단
감속 규칙 'hold', 출발 현실화로 라운드 3 일반 82.6 % / 막다른 공간 39.8 % (run 9보다 일반 +9 %p, 떨림 0.023).
중단 이유: 시뮬레이터 조향 버그 발견 — steer_scale을 최대 조향에도 곱해서 왼쪽 최대가 0.35 rad(실차 0.41)였고,
정책이 "왼쪽을 더 꺾는" 습관을 배우고 있었음. 11/24 수정: 최대 조향은 좌우 0.41 그대로, 중간 영역만 대칭 0.85~1.05.

### IL run 11 (11/24) 설정
run 10 + 조향 수정, a_max 2.5~4.5(측정 전까지), CAPS temporal 0.5, 점수가 비슷하면 떨림이 적은 라운드 선택.
expert(혼자 2랩, 고친 조향): 일반 100 %, 막다른 공간 96.6 %, 최악 조건 87.9 % / 80.7 %.


### IL run 7 (256x2, CAPS 없음, 2026-11-17) — 라운드 3에서 중단
실차 조건 평가 (지연 0.25 s, a_max 3, 노이즈 / 44 트랙 x 3 조건, 2바퀴)

| 라운드 | 8 m/s 중앙선 | 8 m/s 무작위 출발 | 3 m/s 무작위 출발 | 전체 | 처음 보는 트랙 | 조향 떨림 (8/3 m/s) |
|---|---|---|---|---|---|---|
| 0 | 9.1% | 9.1% | 59.1% | 25.8% | 18.2% | 0.008 / 0.008 |
| 1 | 6.8% | 6.8% | 84.1% | 32.6% | 28.8% | 0.016 / 0.024 |
| 2 | 81.8% | 90.9% | 100% | 90.9% | 83.3% | 0.015 / 0.043 |
| 3 | 95.5% | 95.5% | 97.7% | **96.2%** | 93.9% | 0.017 / 0.039 |

- 같은 35개 트랙 2바퀴: 라운드 2 169.6 s → 라운드 3 181.3 s (expert 약 136 s). 갈수록 조심스럽게(느리게) 달림
- 라운드 2 모델 검증: 직선 라인 오차 0.47 m (expert 0.08), 2~5 Hz 조향 떨림 (실차 와리가리 0.4~0.9 Hz와는 다른 미세 떨림)
- 결론: 안정성은 확보, 랩 타임·떨림이 과제 → run 8에서 CAPS + 512x2, 이후 접지력 측정 반영 후 PPO
- 최고 모델: results/imitation/7 (라운드 3). 예전 run 6은 같은 조건 0%
