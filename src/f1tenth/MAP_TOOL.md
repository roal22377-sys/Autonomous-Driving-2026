# 실제 코스 지도 따기 → 시뮬레이터 트랙 → 평가·학습 → 지도 기반 주행 (사용 설명서, 11/25)

11/26(목) 실차 순서는 `CAR_PLAN_1126.md`에 있습니다.

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

### 정답이 있는 시뮬레이션 검증 (11/25, `tools/sim_map_test.py`)
벽 위치를 아는 시뮬레이터 트랙에서 목요일과 같은 지도 주행(IL 11, 상한 2 m/s, 3바퀴)을 하고, 차처럼 기록했습니다.
- LiDAR: 10 m까지, 잡음 0.8 cm, 0.3 % 빠짐.
- odom: 속도 ×1.03, 회전 속도 ×0.85 + 0.02 rad/s 치우침. 11/24 실차 odom처럼 꽃잎 모양이 되는 조건입니다.

그 기록을 지도로 만들고, 다시 트랙으로 바꿔 **정답과 비교**했습니다. 대회 코스(한 바퀴 100 m 이하)와 비슷한 50–90 m 생성 코스(`utils/track_short.py`)와 연습 코스로 했습니다.

| 코스 | 길이 | 루프 클로저 | 궤적 오차 중앙값 | 벽 오차 중앙값 / p90 | 실제 벽 덮음 | 변환 트랙에서 expert |
|---|---|---|---|---|---|---|
| Practice1124 (11/24 실제 코스) | 34 m | 243 | 2.4 cm | 3.3 / 6.6 cm | 97 % | 완주 |
| Rand303 | 61 m | 492 | 5.4 cm | 5.6 / 11.3 cm | 100 % | 완주 |
| Rand300 | 83 m | 618 | 10.5 cm | 7.6 / 15.6 cm | 94 % | 완주 |
| Rand302 | 88 m | 714 | 7.5 cm | 5.9 / 11.6 cm | 100 % | 완주 |
| IMS 축소판 | 90 m | 723 | 6.8 cm | 6.2 / 11.9 cm | 100 % | 완주 |

- **고친 점:** 첫 버전은 짧은 코스에서 루프 클로저를 0–2개밖에 못 찾았습니다. 그래서 벽 오차가 10–63 cm였고(Rand302는 지도가 휘어 63 cm), IMS 축소판에서는 expert가 실패했습니다.
  - 원인은 ICP 정합이 아니었습니다. 정합은 0.6–1.5 cm로 정확했습니다. 문제는 합격 기준인 "스캔 점의 60 % 이상이 맞아야 함"이었습니다. 거친 벽은 직선 법선이 안 잡혀 점이 많이 빠졌습니다.
  - 기준을 **잔차 1.5 cm 미만 + 25 % 이상(최소 80점)**으로 바꿨습니다.
  - 잘못된 클로저는 **최적화 뒤 지도와 25 cm 넘게 어긋나면 버리고 다시 최적화**해서 막습니다.
- **11/24 실제 기록으로 다시 만든 결과:** 스캔 매칭 실패가 52개에서 3개로, 루프 클로저가 176개에서 393개로 바뀌었습니다. 보정량 중앙값은 1.7 m에서 0.45 m로 줄었고, 벽은 한 겹입니다.
- 생성 코스의 남은 오차 5–8 cm는 주로 일부러 거칠게 만든 벽(울퉁불퉁한 장애물) 때문입니다. 매끈한 벽인 연습 코스는 3 cm입니다.
- 다시 돌리기: `tools/sim_map_test.py --workers 3`. 기본값은 연습 코스와 짧은 코스 4개입니다. 짧은 코스를 먼저 만들어야 합니다: `cd src/f1tenth && python -m utils.track_short --random 4 --seeds 3` 그리고 `python -m utils.track_short --maps IMS --seeds 301`. 지도 도구만 바꿨다면 `--reuse_log`로 같은 기록에서 지도만 다시 만듭니다.

---

## 2. 목요일(11/26) 순서

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
tools/map_session.sh --finish Practice1126          # NAME: 영문·숫자만 (_ 없이)
```
- 하는 일: 정지 → bag 가져오기(`logs/run_MMDD_HHMM_extracted`) → `eval_drive.py` → `map_offline.py` → `maps/run_MMDD_HHMM/` → `map_to_track.py` → `src/f1tenth/installation/maps_real/Practice1126(_rev).npz` → `check_track.py` 순서입니다.
- 손으로 옮긴 구간을 빼려면 자율 주행 구간만 씁니다. 시각은 `eval.txt`의 start/dur로 압니다.
  ```bash
  tools/map_session.sh --finish Practice1126 --t0 12 --t1 140
  ```
- 다른 PC에서 기록만 받아 둔 경우는 `python3 tools/map_offline.py logs/<run>_extracted maps/<run>` 다음에 `tools/map_session.sh --rebuild maps/<run> NAME`을 실행합니다.

### 결과 확인 (3가지)
1. `maps/<run>/preview.png`: 오른쪽 패널의 **벽이 한 겹**인지, 빨간 궤적이 바퀴마다 겹치는지 봅니다. `report.txt`의 루프 클로저가 수십 개 이상이고 "fell back to odom"이 키프레임의 10 % 이하이면 정상입니다.
2. `maps_real/<NAME>_check.png`: 흰 영역이 **코스 통로만**인지, 중심선(파랑)과 레이스 라인(빨강)이 통로 안에 있는지 봅니다.
   - 방이나 열린 곳이 붙어 있으면 그 사각형 좌표를 그림에서 읽고 다시 만듭니다.
     ```bash
     tools/map_session.sh --rebuild maps/<run> Practice1126 --block=-10.5,4.75,-7,6.5
     ```
     `--block`은 여러 번 쓸 수 있습니다. 음수로 시작하는 좌표는 `=`로 붙여 씁니다.
3. `check_track.py` 출력: **expert가 모든 상한, 양방향에서 완주**하면 쓸 수 있는 트랙입니다.

### 줄자 확인 (정확도)
통로 폭 3~4곳과 긴 직선 하나를 줄자로 재서 `_check.png`의 폭과 비교합니다(목표 오차 ±5 cm). 차이가 크면 결과와 함께 알려 주세요.

### 저장
```bash
git add maps/<run> src/f1tenth/installation/maps_real/Practice1126*
git add -f logs/<run>_extracted          # logs/는 .gitignore에 있음
git commit -m "map: practice course 11/26" && git push
```

---

## 3. 평가와 학습에 쓰기

| 용도 | 명령 | 비고 |
|---|---|---|
| 모델 비교에 실제 코스 열 추가 | (src/f1tenth에서) `python -u compare_models.py ppo_s2r/5 ppo_s2r/12190 --seeds 2 --extra_tracks installation/maps_real/Practice1126.npz installation/maps_real/Practice1126_rev.npz` | 상한 8·4·2, 출발선에서 2랩. "real course" 열과 상한 8의 2랩 시간 |
| 실차 전에 한 트랙만 빨리 확인 | `python3 tools/check_track.py src/f1tenth/installation/maps_real/Practice1126.npz --models ppo_s2r/12190 --caps 2 3 4 --seeds 3` | 상한마다 완주 수, 2랩 시간, 실패 시각 |
| PPO 평가마다 실제 코스 보고 | `train_ppo_s2r.py ... --real_tracks installation/maps_real/Practice1126.npz installation/maps_real/Practice1126_rev.npz` | 저장 기준에는 안 들어가고 기록만 합니다(`[real course @ update N]`, `train_log.pkl`의 `real_success`) |
| **그 코스로 학습** (대회 코스용) | 위 옵션에 `--real_train_share 0.1` 추가 | 에피소드의 10 %를 그 코스에서 학습합니다. 연습 코스는 대회 코스와 다르니 **평가만** 하고, 대회 코스를 따면 이 옵션으로 짧게 미세조정합니다 |

- 실제 코스 트랙은 `installation/maps_real/`에 따로 둡니다. 생성 트랙 학습 목록(`maps_gen`)에 섞이지 않아서, 옵션을 주지 않으면 학습에 쓰이지 않습니다.
- 트랙 파일에 통로 폭(`w_left`/`w_right`)이 있어서 막다른 공간 벌점(`--bay_cost`)도 그대로 작동합니다.

---

## 4. 지도 기반 주행 (계획 B, 11/25 결정·진행 중)

지도와 차의 위치를 알면 LiDAR 10 m 너머의 코스를 미리 알 수 있습니다. 그러면 직선에서 더 빨리 달리고 늦게 제동할 수 있습니다. 11/25 분석에서 랩 손해의 70 %가 직선 최고 속도였습니다(PPO 5.3 m/s, expert 7.3 m/s).

**지도를 만들 때는 SLAM**(스캔 매칭 + 루프 클로저 + 포즈 그래프, 이 문서의 `map_offline.py`)을 씁니다. **달릴 때는 지도를 고정하고 위치만 추정**합니다(localization). 그 위치로 앞 코스 정보를 계산해 정책에 넣습니다.

| 부분 | 파일 | 내용 |
|---|---|---|
| 정책 입력 (15개) | `utils/preview.py`, `configs/task/f1tenth_preview.yaml` (obs 114 → 129) | 레이스 라인에서 벗어난 거리와 방향 오차, 앞 2·4·6·8·10·15·20·30 m의 곡률, 앞 0·5·10·20 m의 목표 속도(명목 차량의 속도 계획, 속도 상한 적용), 위치 유효 표시. 좌표가 아니라 상대량이라 코스가 달라도 의미가 같습니다 |
| 시뮬레이터 | `utils/env.py` (`track_preview`) | 위치 추정 오차를 흉내 냅니다: 천천히 떠도는 오프셋 5 cm·1°, 떨림 1 cm, 학습 때는 가끔 0.3 m 튐과 0.5–3 s 위치 끊김(에피소드의 15 %). 끊기면 유효 표시가 0이고 나머지 입력도 0이어서, 정책이 **LiDAR만으로 계속 달리도록** 배웁니다 |
| 실차 위치 추정 | `utils/localizer.py` (`ScanLocalizer`, `CarPreview`) | 스캔마다 바퀴 odom으로 예측한 뒤, 지도 벽에 점-선 ICP로 맞춥니다(빔 3개 중 1개, 잔차 3 cm 미만, 60점 이상 맞으면 채택). 0.2 s 동안 못 맞추면 유효 표시를 0으로 하고 넓게 다시 찾습니다. 시작 위치는 첫 스캔에서 코스를 따라 0.5 m마다 양방향으로 찾습니다. 차가 반대로 서 있으면 `_rev` 코스를 씁니다 |
| 차 노드 | `agent_node.py` (`course`, `map_dir`, `laser_x`, `loc_stride` 파라미터), `config/agent_preview.yaml` | 모델 설정에 `track_preview`가 있을 때만 켜집니다(기존 모델은 그대로). `/loc_pose`를 발행하고, bag에 기록합니다(frame_id `map`/`map_lost`, z = ICP 잔차) |
| 학습 | `train_il_s2r.py --task_config configs/task/f1tenth_preview.yaml --short_share 0.5 --short_eval 1` | IL 13부터. 짧은 코스(50–100 m, `utils/track_short.py`, `installation/maps_short`)를 에피소드의 50 %로 학습하고, 따로 둔 짧은 코스 16개로 평가해 점수에 넣습니다 |

**검증 (11/25)**

| 시험 | 결과 |
|---|---|
| `tools/loc_test.py`, 11/24 실제 기록(4 m/s, 62 s) / 같은 날 지도 | 시작 위치 자동 탐색 0.3 s. 2475 스캔 모두 유효. 지도 도구 포즈와의 차이 중앙값 2.9 cm, p95 13 cm. 방향 0.6°. 스캔당 2.1 ms(PC) |
| `tools/loc_test.py --sim`, 시뮬레이터 정답 비교: 상한 2로 딴 지도에서 상한 4–5로 다시 주행 | 연습 코스 2.8 cm(p95 5.8), Rand300 10.6 cm(지도 자체 오차 10.5 cm), IMS 축소판 6.6 cm. 모두 100 % 유효 |
| `CarPreview` (agent_node와 같은 경로), 11/24 기록 | 정방향과 `_rev` 코스 모두 시작 성공(`_rev`는 자동 전환), 미리보기 15개, 스캔당 2.9 ms |
| `tools/agent_replay.py` (ROS 없이 agent_node 전체를 기록으로 재생) | 파라미터, 모델 로드, 관측 크기, 위치 추정, 명령 발행까지 확인 |
| `tools/loc_test.py --stress` (일부러 망가뜨리기: LiDAR 1.5 s 끊김, 위치를 1.2 m·20° / 6 m·115° 강제로 옮김), 실제 기록 1개 + 시뮬레이터 3개 | LiDAR가 돌아오면 0.2–3.5 s 안에 다시 찾음(한 코스는 6 m 옮김 뒤 남은 24 s 동안 못 찾음 → 그동안 LiDAR만으로 주행). **틀린 위치를 유효로 표시한 스캔은 옮긴 직후 7–8개(0.2 s, 짧은 끊김을 odom으로 넘기는 구간)뿐**. 위치를 잃은 동안에도 스캔당 6.6 ms 이하(PC) |

**11/25 저녁 점검에서 고친 것 (위치 추정)**
1. 위치를 잃었을 때 다시 찾는 계산이 스캔 하나에서 19 ms(PC)였습니다. Jetson이면 40 Hz 제어가 느려질 수 있어서, 스캔마다 한 번만 시도하도록 나눴습니다.
2. 6 m 옮긴 시험에서 코스의 비슷한 곳에 잘못 붙고도 "유효"로 표시했습니다. 고친 내용은 다음과 같습니다.
   - 스캔이 지도 벽에 얼마나 맞는지(10 cm 안 비율)를 계속 봅니다. 맞는 위치는 0.58 아래로 내려간 적이 없고, 틀린 위치는 중앙값 0.33이었습니다. 10스캔 평균이 0.45 아래로 내려가면 잃은 것으로 처리합니다.
   - 다시 찾은 위치는 0.8 이상 맞아야 하고, 그 뒤 0.2–1 s 동안 **코너처럼 앞뒤·좌우가 모두 잡히는 곳에서** 계속 맞아야 유효로 칩니다. 직선에서는 몇 m 앞뒤가 똑같아 보이기 때문입니다.
   - 시작할 때 1.5 m 넘게 떨어진 다른 위치도 거의 똑같이 맞으면 출발하지 않고 "코너가 보이는 곳으로 옮기라"고 알립니다.

**대비책:** 위치 추정이 불안하면 LiDAR 전용 모델(`agent_ilppo.yaml`)로 돌아갑니다.

**대회 주간:** 대회 코스를 2일 전에 땁니다(이 문서 2절과 같은 흐름) → `train_ppo_s2r.py --real_tracks ... --real_train_share 0.1~0.3`으로 그 코스에서 미세조정 → 전날 실차 확인 순서입니다.

---

## 5. 도구 설명

| 도구 | 하는 일 |
|---|---|
| `tools/map_session.sh` | 차 세션(= `MODEL=mapping tools/drive_session.sh`). `--finish NAME`은 기록 → 지도 → 트랙 → 확인까지, `--rebuild`는 트랙만 다시 만듭니다 |
| `tools/map_offline.py LOG OUT [--t0 --t1 --res --laser_x --loop_res --loop_inliers --loop_prune]` | 오프라인 2D 그래프 SLAM입니다.<br>① 키프레임(0.15 m / 5°)마다 1081개 빔을 점-선 ICP로 주변 지도에 맞춤(odom을 초기값으로)<br>② 루프 클로저 A: 모양이 뚜렷한 키프레임(코너 등)을 쌓인 오차 반경(이동 거리 × 4 %, 최대 12 m) 안에서 찾아 맞춤<br>③ 루프 클로저 B: 보정된 궤적 기준 1.2 m 안의 재방문을 모두 맞춤(반복)<br>④ 합격: 잔차 1.5 cm 미만, 스캔의 25 % 이상(80점 이상), 방향이 퇴화하지 않음. 포즈 그래프 최소제곱(soft_l1) 뒤 25 cm 넘게 어긋나는 클로저는 버리고 다시 최적화<br>⑤ 광선 추적으로 점유 격자를 만들어 `map.pgm`/`map.yaml`(ROS map_server 형식)로 저장. `poses.csv`, `preview.png`, `report.txt`도 남김 |
| `tools/sim_map_test.py [TRACK ...] [--out --laps --cap --workers --reuse_log]` | 정답 비교 검증(위 표). 시뮬레이터 지도 주행 → 차와 같은 잡음의 기록 → `map_offline.py` → `map_to_track.py` → 궤적·벽·폭 오차와 expert 완주 → `report.md`, `compare.png` |
| `tools/map_to_track.py MAP --name NAME [--block --max_half 2.5 --close_gaps 0.3]` | 지도와 궤적으로 트랙을 만듭니다.<br>① 궤적과 이어진 빈 칸(경로에서 2.5 m 안, 벽의 0.6 m 이하 틈은 닫음)<br>② 한 바퀴 궤적을 좌우 벽 사이 가운데로 4번 다시 맞춘 중심선(0.1 m 간격)<br>③ 좌우 폭(1.5 m 중앙값 + 0.2 m 넘는 오목한 곳은 통로 폭에서 제외)<br>④ 생성 트랙과 같은 레이스 라인 최적화<br>⑤ 정방향과 역방향 `.npz`(`utils/track_gen.to_track` 형식) |
| `tools/check_track.py TRACK [--models --caps --seeds --workers]` | 시뮬레이터에서 expert와 모델을 그 트랙에서 달려 완주 수, 2랩 시간, 실패 시각을 보여 주고 `_drive.png`를 만듭니다 |
| `tools/loc_test.py LOG MAP [--track --truth --sim --map_truth --stride]` | 기록으로 위치 추정 시험(차와 같은 코드): 유효 비율, 끊김, 위치·방향 오차, 미리보기 입력 오차, 스캔당 시간, `loc_test.png` |
| `tools/agent_replay.py LOG [--config --set k=v]` | ROS 없이 `agent_node`를 기록 위에서 그대로 실행(열린 루프). 차 세션 전에 설정·모델·위치 추정 오류를 잡습니다 |
| `tools/check_short_tracks.py [WORKERS] [--remove_failed]` | `installation/maps_short`의 짧은 코스를 expert로 달려 보고, 완주 못 하는 코스를 지웁니다(11/25: 114개 중 8개 제거) |
| `tools/log_map.py` | (분석용) 사고 구간 지도. 루프 클로저가 없어서 트랙으로는 쓰지 않습니다 |

**트랙 파일 형식**(`utils/track_gen.py`와 같음): `occupancy`(행 0 = 최소 y, >128 빈 칸), `resolution`, `origin`, `centerline`(0.1 m 간격, 주행 방향),
`w_left`/`w_right`, `raceline`, `length`, `runoffs`(실제 코스는 빈 배열), `name`, `map`, `reverse`, `original=True`, `gen_sig='real:<NAME>'`.

**선택: 차에서 실시간 지도 (slam_toolbox).** 위 흐름에는 필요 없습니다. 원하면 차에 `sudo apt install ros-foxy-slam-toolbox ros-foxy-nav2-map-server`를 설치하고, bringup 중에 `ros2 launch slam_toolbox online_async_launch.py`(인자 이름은 Foxy 버전에서 확인)를 실행한 뒤 `ros2 run nav2_map_server map_saver_cli -f ~/maps/<name>`으로 저장합니다.
이 경우 궤적(`poses.csv`)이 따로 없습니다. 그래서 `map_to_track.py`에는 같은 기록으로 `map_offline.py`를 돌려 얻은 `poses.csv`를 `--poses`로 줍니다. 두 지도는 좌표계가 다르므로 `map_offline.py` 지도를 쓰는 쪽을 권합니다.
