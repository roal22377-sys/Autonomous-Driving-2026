# run_1126_1741: speed-ramp grip test (preview PPO 21 u30 + LiDAR fallback), cap 4.0 -> +0.5 / 15 s, Practice1126b

- 68.5 s / 212 m, then wall contact at cap 6.0 (right hairpin exit, outer wall, x 0.6 y 4.4).
- Cause: VESC fault 4 (ABS over current) at 114.7 s, 5.5 m/s, the moment the model started braking for the right hairpin
  (cmd 5.43 -> 1.26 m/s in 1 s); the car coasted 5.5 -> 4.8 m/s into the hairpin. Same as run_1126_1725 (cap 8).
- Lateral acceleration (localized yaw rate x odom speed, p95; p99/max are polluted by pose jumps):
  cap 4.0: 4.97, 4.5: 4.75, 5.0: 5.12, 5.5: 5.44 m/s^2 -> still rising, no plateau: grip >= ~5.4 m/s^2 (mu >= 0.55).
  The race-line speed plan uses a_lat 3.5 m/s^2.
- Localization 87.3 % valid while driving, 2 lost stretches (longest 5.8 s, around the crash).
