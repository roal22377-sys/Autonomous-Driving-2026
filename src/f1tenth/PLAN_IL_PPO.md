# 학습 방식 계획: IL → PPO fine-tuning

작성일: 2026-11-12

## 결론
Imitation Learning(IL)으로 학습한 정책을 초기값으로 두고, PPO로 fine-tuning해서 랩타임을 줄인다.

- **IL**: 완주 가능한 정책을 빠르게 확보한다. run 6으로 완료(시뮬레이션 평가 33/33 랩, expert 대비 랩타임 약 1초 차이).
- **PPO**: IL 정책을 이어받아 expert(`CurvaturePursuitPlanner`)보다 빠른 주행을 학습한다.
- 아직 PPO fine-tuning을 실험하기 전이라 개선 여부는 검증되지 않았다.

## 근거
1. **IL 단독의 한계**: `algorithm/imitation/agent.py`가 expert 행동을 MSE로 따라 하는 behavior cloning이라 성능 상한이 expert다. 리더보드(`leaderboard.py`)는 완주율과 평균 랩타임을 보는데, 완주율은 이미 충분하므로 남은 점수는 랩타임에서 나온다.
2. **RL 단독의 위험**: 관측(LiDAR 54개 + 속도), 2.5M 스텝, 23개 맵 조건에서 처음부터 학습하면 벽에 부딪히는 단계에서 오래 머문다. 시뮬레이터 dynamics에 과적합되기 쉬워 sim2real에서도 불리하다.
3. **SAC보다 PPO가 맞는 이유**: IL과 PPO는 같은 `Actor` 클래스(`algorithm/common/models.py`: LeakyReLU, hidden 256, tanh mean)를 쓰므로 가중치를 그대로 불러올 수 있다. SAC는 `SquashedActor`(Mish, log_std head)라서 구조가 달라 바로 옮길 수 없다.

## 구현 시 주의점
1. **IL 가중치와 `obs_rms` 로드**: `results/imitation/6`에서 불러온다. fine-tuning 동안 `obs_rms`는 고정한다.
2. **critic 먼저 학습**: 처음 몇 번의 업데이트는 actor를 고정하고 critic만 학습해서, 초반에 IL 정책이 무너지는 것을 막는다.
3. **BC 정규화 항**: `actor_loss + λ·MSE(π(s), expert action)` 형태로 넣고 λ를 점점 줄인다.
4. **탐색 noise 축소**: `log_std_init`을 -1.0(std 0.37)에서 -2 안팎으로 줄인다.
5. **에피소드 길이와 reward**: `max_episode_steps`를 한 바퀴가 들어갈 만큼 늘린다(현재 1000스텝 = 25초). 공격적으로 달리다 완주율이 떨어지지 않도록 충돌 페널티(`COLLISION_COST`)를 키우는 것을 검토한다.
6. **sim2real 대비**: `obs_noise=True`를 켠다. 가능하면 `mu`나 조향 지연에 약한 domain randomization을 넣는다.

## 진행 순서
1. 실차에는 IL run 6을 백업 모델로 유지한다(이미 `agent_node`에 배포됨).
2. `main_rl.py`에 `--pretrained imitation/6` 옵션을 추가하고, 위 주의점을 반영해 PPO fine-tuning을 구현한다.
3. `leaderboard.py`로 IL과 PPO-finetune을 비교한다. 완주율이 유지되면서 랩타임이 줄었을 때만 교체한다.
