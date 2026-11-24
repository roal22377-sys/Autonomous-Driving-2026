# PPO 5 (11/24): IL run 11 + PPO, update 100 — first PPO model that passed every save gate

Car: `MODEL=ilppo tools/drive_session.sh` (`config/agent_ilppo.yaml`: name ppo_s2r, algo_idx 5). Start at speed_limit 2.0, then 4.0.

| simulator evaluation (2 laps, simulator before the 11/24 sync) | IL 11 | PPO 5 update 100 |
|---|---|---|
| eval tracks | 87.9 % | **93.2 %** |
| - held-out variant w105 | 76 % | 86 % |
| worst case (delay 0.2 s, mu 0.4, accel 2.5, ...) | 83.3 % | **89.4 %** |
| open-space (big dead-end bays) | 67.0 % | 67.0 % |
| 2-lap time on runs both finished | reference (199 s) | **28.5 s faster (~14 %)** |
| steering jitter | 0.024 rad | **0.012 rad** |

- Training: PPO 4 update 40 -> PPO 5 with 25 % hard-condition episodes and 20 % open-space training tracks,
  adaptive collision cost (ratchet, max 10000). Evaluations after update 100 were 65-79 %: update 100 may be a good draw.
- Steering conversion (backup/f1tenth.yaml) is IL 11's: max_steer 0.41, command gains 1.077 / 1.212 — the same mapping
  IL 11 drove without contact on 11/24 with the calibrated car (servo offset 0.567, gain 3100), so the two compare fairly.
- Trained before the 11/24 simulator sync (mu 0.5, full lock 0.41). IL 12 / PPO 7 include the 11/24 values.
