IL run 13 (2026-11-25, plan B): lidar 88 bins + speed + action history + speed limit + **track preview** (obs 129:
race-line offset / heading error, curvature 2-30 m ahead, speed plan 0-20 m ahead, localization-valid flag;
utils/preview.py, configs/task/f1tenth_preview.yaml), simulated localization error (5 cm / 1 deg wander, jumps,
0.5-3 s losses in 15 % of the training episodes). 11/24 car calibration + separate accel / brake limits (2.9 / 3.7).
Training tracks: generated F1 variants + open-space + short 50-100 m courses (utils/track_short.py) in 50 % of the
episodes (--short_share 0.5). Stopped by Claude after round 9 (score near the ceiling; the night goes to PPO 14).

Saved = round 9 (2 laps):
  eval tracks 100 % (cap 8 centerline / cap 5 centerline / cap 8 mild random), open-space big 95.5 %,
  held-out short courses 100 % (2-lap 32.1 s), score 98.5, steering jitter 0.0041 rad, uncapped 2-lap 132.5 s.
  Rounds (eval / open / short): r0 61.4/56.8/71.9, r2 88.6/79.5/96.9, r4 94.7/87.5/100, r7 98.5/93.2/96.9, r9 100/95.5/100.

Weak WITHOUT localization (preview valid = 0 for the whole run): eval tracks cap 8 36 %, short courses 69 %,
cap 3: 68 % / 81 %, cap 5: 77 % / 69 %. On the car agent_preview.yaml stops the car when the localization is lost
for > 0.3 s (lost_stop). PPO 14 trains with longer losses and 10 % episodes without localization.
Car: MODEL=preview tools/drive_session.sh (config/agent_preview.yaml; course / map_dir from tools/map_session.sh).
