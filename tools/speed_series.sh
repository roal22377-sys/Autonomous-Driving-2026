#!/bin/bash
# Straight-line speed tests toward a flat wall, one speed after another (bringup running, no agent).
# Started by tools/drive_session.sh --speed in its last tab; or by hand on the car:
#   tools/speed_series.sh              # 2.0 2.5 3.0 m/s (about 6-10 m of straight floor)
#   tools/speed_series.sh 2 3 4        # chosen speeds (4 m/s needs ~13 m)
# Each speed: plan (distance needed) -> Enter = drive now (3 s countdown), s = skip, q = quit.
# Analysis on the PC: tools/drive_session.sh --finish --speed
SPEEDS=("$@")
[ ${#SPEEDS[@]} -gt 0 ] || SPEEDS=(2.0 2.5 3.0)
cd ~/ros2_ws
for v in "${SPEEDS[@]}"; do
  echo
  echo "================ speed $v m/s ================"
  python3 tools/speed_test.py --speed "$v" --dry_run
  echo ">>> Put the car facing a flat wall at least that far away, straight. TELEOP tab: r (released)."
  read -p ">>> Enter = drive now (3 s countdown), s = skip, q = quit: " ans
  case "$ans" in
    q) break ;;
    s) continue ;;
  esac
  python3 tools/speed_test.py --speed "$v" --no_prompt || true
done
echo "[done] speed series finished. On the PC: tools/drive_session.sh --finish --speed"
