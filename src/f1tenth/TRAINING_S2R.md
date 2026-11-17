# sim2real 학습 파이프라인 (IL → PPO)

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
| PPO | IL 초기화, 비대칭 critic(시뮬레이터 정답 상태 14개), critic 먼저 학습, IL 쪽 BC 항 감소, 진행거리 보상 + 충돌 −200 + 조향 급변 벌점, IL보다 좋을 때만 저장 | `train_ppo_s2r.py` |

## 실행 (src/f1tenth에서)

```bash
# 0. 트랙 생성 (한 번, 약 3분)
OMP_NUM_THREADS=1 ~/venv/torch210/bin/python3 -m utils.track_gen 6 22

# 1. IL
OMP_NUM_THREADS=1 ~/venv/torch210/bin/python3 -u train_il_s2r.py --algo_idx 7 --workers 22 \
    --expert_steps 1.5e6 --dagger_iters 3 --dagger_steps 8e5 --dart_std 0.08 --grad_steps 20000

# 2. PPO 미세조정
OMP_NUM_THREADS=1 ~/venv/torch210/bin/python3 -u train_ppo_s2r.py --il imitation/7 --algo_idx 1 --workers 22

# 3. 비교 (같은 트랙·출발 위치, 실차 조건 eval / 원래 시뮬레이터 ideal, 속도 상한 8·3)
OMP_NUM_THREADS=1 ~/venv/torch210/bin/python3 -u evaluate_s2r.py imitation/6 imitation/7 ppo_s2r/1
```

기준 (IL run 6, 실차 조건): 상한 없음 0%, 3 m/s 33% 완주 (트랙 3개 샘플). 원래 시뮬레이터에서는 100%.
