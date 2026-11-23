IL run 11 (2026-11-24): lidar 88 bins, 11/23-24 calibration (steering lock 0.41 rad both sides, symmetric
mid-range scale, a_max 2.5-4.5, mu 0.5 x 0.8-1.25), expert 'hold' rule, realistic starts, CAPS 0.5/0.3.
Stopped after round 10 (3 rounds without improvement). Saved = round 7:
  eval tracks 2-lap success 87.9 % (cap 8 centerline / cap 5 centerline / cap 8 mild random), open-space big 67.0 %,
  score 77.5, steering jitter 0.024 rad, uncapped 2-lap ~199 s (expert ~156 s).
Below the PPO start gate (98 % / 95 %): PPO with --force_start (11/23 decision: IL plateaued -> improve with PPO).
Known: still steers too little where large steering is needed (31-43 % of the expert's angle above 0.2 rad).
Car: MODEL=il tools/drive_session.sh (config/agent_il.yaml -> imitation/11).
Methods: src/f1tenth/IL_METHODS_1124.md
