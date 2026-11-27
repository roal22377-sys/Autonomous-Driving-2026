# Shared-scan close-traffic pretraining audit, 2026-11-28

The frozen primary actor and the actual scan localizer were evaluated on the Practice1126b nominal map with a scripted car starting 0.2 lap ahead. The opponent follows the same loop at a 2.0 m/s cap. Each ego policy receives the same LiDAR scan as its localizer. The sensor-only braking envelope is an offline diagnostic, not a trained or deployed policy. No IL/PPO update occurred.

| Seed | No opponent, raw | No opponent, brake | Close opponent, raw | Close opponent, brake |
| --- | ---: | ---: | --- | ---: |
| 7 | 16.575 s, complete | 17.300 s, complete | 4.650 s, opponent collision | 27.950 s, complete |
| 17 | 16.700 s, complete | 17.425 s, complete | 4.975 s, wall collision | 27.975 s, complete |
| 27 | 16.200 s, complete | 17.025 s, complete | 3.100 s, opponent collision | 28.125 s, complete |

The raw actor completed 0/3 close-traffic runs. The envelope completed 3/3, but added 0.758 s mean to the clean two-lap runs (paired). Its traffic runs took about 28 s, so treating the envelope as a final racing solution would sacrifice substantial pace. The localizer reported no false-valid scan in these runs; the raw close-traffic failures happened even when localization was correct. This separates immediate collision avoidance from a localization-loss failure. The wall collision at seed 17 also shows that avoidance and path tracking interact.

These are three deterministic seeds on one known map with a scripted, nonreactive opponent and favorable exact initial pose/map. The envelope is not qualified for physical use or for all relative gaps. The next learning stage should preserve the clean-track pace evaluation, add opponent-relative information derived from available LiDAR and history without privileged true position, and train/evaluate different gaps and speeds. Evaluate scan-localizer loss/reacquisition and physical wall drift separately with the nominal mapped reference held fixed. Opponent completion and clean pace must both be promotion gates.

For a separate map-mismatch check, the raw frozen policy completed both laps on all three paired seeds with the Practice physical wall warped by 0.05 m, and again on all three with a 0.10 m warp. The stored nominal map/reference stayed unchanged and the localizer used the physical shared scan. Times were 16.575/16.650/16.225 s at 0.05 m and 16.550/16.675/16.175 s at 0.10 m for seeds 7/17/27; unwarped times were 16.575/16.700/16.200 s. No false-valid localization was recorded. This specific smooth warp did not expose a clean-run failure; it does not establish robustness to arbitrary barrier displacement, other tracks, or simultaneous traffic and localization loss. The existing combined 0.10 m warp and close-traffic seed-7 trial still failed raw and completed with braking.

Ignored raw reports: `src/f1tenth/results/traffic_localizer_1127/{combined_hard_off,combined_hard_brake,control_seed7,control_brake,close_seed17_raw_1128,close_seed17_brake_1128,clean_seed17_raw_1128,clean_seed17_brake_1128,close_seed27_raw_1128,close_seed27_brake_1128,clean_seed27_raw_1128,clean_seed27_brake_1128}/report.json`.
