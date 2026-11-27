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

An isolation run on the same seed replaced geometry alone, geometry plus optimized speed, or geometry plus optimized curvature. All six nominal/worst teacher runs crashed within 1.90-2.30 s. The shape and its narrow wall clearance are therefore the immediate blocker; replacing only speed or curvature does not rescue this reference.

## Existing clearance contract, missed by the joint optimizer

The stored regular-map generator sets `RACELINE_MARGIN = 0.55` m center-to-wall (0.15 m half body plus 0.40 m tracking allowance). The hall-map generator uses `HALL_MARGIN = 0.30` m plus `TRACK_ERR = 0.15` m where corridor width allows (nominal 0.30 m body-to-wall allowance). Narrow corridors can have a smaller physically possible local margin. The experimental joint optimizer used only 0.03 m sampled body clearance plus a small reserve; this did **not** preserve the original generator's safety contract.

Independent dense raster body sampling of the five known representatives gave:

| Map family | Stored minimum body clearance | Joint minimum body clearance |
| --- | ---: | ---: |
| Practice | 0.300 m | 0.150 m |
| 2025 example | 0.320 m | 0.080 m |
| Regular | 0.358 m | 0.113 m |
| Open | 0.320 m | 0.113 m |
| Short, narrow exception | 0.160 m | 0.113 m |

All five new lines reduce clearance. The short case illustrates why one global 0.30 m floor cannot be imposed on every map. A 0.18 m body-clearance sensitivity solve was stopped when this mismatch was identified; its output is not a candidate. The next 2025 example trial uses the previously established 0.30 m body allowance, and its result must pass dense raster and closed-loop checks before any promotion. For arbitrary maps, use the map family's existing nominal safety margin with a width-limited local exception; audit every fallback explicitly. Do not infer tire grip from slow real-car runs or assume that sampled static clearance guarantees dynamic tracking.

The independent source-relative raster audit (`relative_clearance_audit_1128.json`) rejects all five old candidates, with up to half a grid cell of measurement tolerance. This is a conservative promotion screen, not a proof of dynamic safety. The original line stays the fallback for every rejected map.

## Existing-margin 2025 pilot

`course2025_existing_margin_v9` reoptimized the same known 2025 example at 6 m/s² nominal lateral-acceleration hypothesis and 0.30 m sampled body allowance. The optimizer returned a feasible iterate after its 50-iteration limit; the dense spline audit found 0.314 m minimum body clearance. The independent raster screen passes the source-relative margin gate. Its quasi-steady model time is 14.224 s per lap versus 13.211 s for the unsafe earlier candidate. This is a planning prediction, not a measured lap.

In paired closed-loop seed 7/17/27 two-lap runs, the **joint geometry+speed+curvature package** completed 3/3 nominal and 3/3 worst for both the frozen policy and privileged teacher. Mean completed-run times were policy 33.725/37.317 s and teacher 32.308/35.567 s (nominal/worst). The stored source had policy 33.458 s on 3/3 nominal and 37.125 s on only 1/3 worst; teacher 32.108/35.450 s on 3/3 each. Small samples do not establish a success gain, and the new package is slower in the nominal comparisons. Geometry alone gave policy 31.825 s on 3/3 nominal but the teacher crashed 3/3 nominal; this cannot be promoted as a training teacher reference. More paired seeds and different map families remain required.

Seven additional seeds (37-97) bring the same known-map comparison to ten seeds. The frozen policy completed 10/10 nominal on both references, 4/10 worst on stored source, and 10/10 worst on the joint package. Its nominal mean was 33.417 s stored versus 33.680 s joint (+0.262 s paired). Among the four worst cases completed by both references, joint was +0.250 s slower. The teacher completed 10/10 in both modes on both references; joint was +0.170/+0.122 s slower paired (nominal/worst). This supports a *known-map robustness hypothesis*, not a general win or faster laps. No optimized reference is promoted until other map families and uncertainty cases pass.

## Practice course: clearance is necessary but insufficient

The Practice1126b existing-margin pilot requested 0.28 m body clearance and passed the static source-relative gate (stored 0.300 m; optimized 0.320 m). However, after the 50-iteration limit its best feasible speed profile averaged only 0.86 m/s; the quasi-steady lap-time estimate was 34.912 s on a roughly 30 m path. This is a solver quality failure, even though all sampled feasibility constraints passed. On paired seeds 7/17/27, the stored source policy completed 3/3 nominal and worst at 16.608/18.667 s mean two-lap time. The optimized package policy completed only 2/3 each at 27.537/27.487 s among completions; the teacher completed 0/3 in both modes. Reject this candidate. A planner output must clear both feasibility and a source-relative time/closed-loop quality gate before becoming a learning reference.

The joint planner's quasi-steady geometric/energy feasibility checks are insufficient to qualify a new line for the delayed, slipping closed-loop vehicle. Keep stored references and the current learning checkpoint intact. Next, inspect first-corner command, pose, tire/steering response, and wall-clearance margins; test a startup merge and speed envelope on representative maps before any broad training-reference replacement. Require clean and worst-mode completion across track families, then independent seeds/geometries. The 2025 course is an example, not the future competition map.
