# Pretraining reference tracking diagnostic (2026-11-28)

Status: **not ready for learning**. All runs below use a frozen actor or an offline privileged teacher. No IL/PPO update, real-car command, or reference promotion occurred.

## Five-family curvature feedforward trial

The experimental `feedforward_feedback` teacher was tested on one known representative per family, in nominal and randomized worst modes, with the episode-physics speed adaptation and heading-merge gate enabled. It completed both modes on Practice only. On Course2025 it completed the stored source reference but crashed on the jointly optimized reference after 2.52/6.78 s. On regular and open, it crashed even on the stored source reference; on short it crashed on the nominal source reference. This controller trial is rejected. Individual results and traces are in `src/f1tenth/results/joint_mintime_1127/*/ff_feedback_1128.json` (ignored local artifacts).

## 2025 example map: off-line lookahead trial

An offline-only `lad_off_gain` of 1.0 or 2.0 m/m was added to the existing teacher; the production/default value remains unchanged. Each trial used one seed, both nominal and randomized worst modes, stored source versus jointly optimized reference, and the same episode-physics speed adaptation and heading-merge gate.

| Reference | Condition | Gain 1.0 | Gain 2.0 |
| --- | --- | ---: | ---: |
| Stored source | Nominal | Complete, 32.275 s | Complete, 32.475 s |
| Stored source | Worst | Complete, 35.800 s | Complete, 36.075 s |
| Jointly optimized | Nominal | Crash, 2.275 s | Crash, 2.275 s |
| Jointly optimized | Worst | Crash, 1.950 s | Crash, 1.950 s |

On the nominal run, initial centerline spawn distance to the new line was about 0.59 m versus 0.26 m to the stored line. The new-line run reached about 0.63 m maximum line error in the first 0.5 s. The larger lookahead did not resolve the failure. These are descriptive traces, not a causal attribution of the crash to spawn offset alone.

The joint planner's quasi-steady geometric/energy feasibility checks are insufficient to qualify a new line for the delayed, slipping closed-loop vehicle. Keep stored references and the current learning checkpoint intact. Next, inspect first-corner command, pose, tire/steering response, and wall-clearance margins; test a startup merge and speed envelope on representative maps before any broad training-reference replacement. Require clean and worst-mode completion across track families, then independent seeds/geometries. The 2025 course is an example, not the future competition map.
