#!/bin/bash
# Circle tests for several steering commands, one after another (bringup must be running, no agent).
# Started by tools/drive_session.sh --circle in its last tab; or by hand on the car:
#   tools/circle_series.sh                                  # default commands below
#   tools/circle_series.sh 0.44 -0.5 -- --speeds 2 2.5 3    # chosen commands; options after -- go to circle_test.py
# Each command: dry-run (circle sizes) -> Enter = run, s = skip, q = quit.
# The keyboard tab still overrides (s = stop). Analysis: tools/drive_session.sh --finish --circle
STEERS=() OPTS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; OPTS=("$@"); break; fi
  STEERS+=("$1"); shift
done
# full lock both sides (grip shows up at low speed), then mid and small (steering map)
[ ${#STEERS[@]} -gt 0 ] || STEERS=(0.44 -0.5 0.3 -0.3 0.15 -0.15)

cd ~/ros2_ws
for s in "${STEERS[@]}"; do
  echo
  echo "================ steering $s ================"
  python3 tools/circle_test.py --steer "$s" --dry_run "${OPTS[@]}"
  read -p "Enter = run, s = skip, q = quit: " ans
  case "$ans" in
    q) break ;;
    s) continue ;;
  esac
  python3 tools/circle_test.py --steer "$s" "${OPTS[@]}" || true
done
echo "[done] circle series finished. On the PC: tools/drive_session.sh --finish --circle"
