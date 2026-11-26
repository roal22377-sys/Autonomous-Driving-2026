#!/bin/bash
# Map a real course: the car drives itself slowly for a few laps while the bag records, then the map is built
# offline with loop closure and turned into a simulator track (src/f1tenth/MAP_TOOL.md).
#
#   tools/map_session.sh [--no-deploy]          # = MODEL=mapping tools/drive_session.sh (6 tabs: bringup / TELEOP
#                                               #   e-stop / bag / status / cmd monitor / agent with agent_mapping.yaml:
#                                               #   IL 11 at 2.0 m/s). Put the car on the course, press s in
#                                               #   TELEOP, Enter in the agent tab, r in TELEOP -> it drives.
#                                               #   Let it do 3+ laps (s = stop at any time), then s.
#   tools/map_session.sh --finish NAME [--t0 S --t1 S]
#                                               # stop + fetch the bag + tools/map_offline.py -> maps/<run>/
#                                               #   (map.pgm/.yaml, preview.png, report.txt) + tools/map_to_track.py
#                                               #   -> src/f1tenth/installation/maps_real/NAME(_rev).npz + _check.png
#                                               #   + tools/check_track.py (expert / PPO 12 drive it in the simulator)
#   tools/map_session.sh --rebuild maps/<run> NAME [--block x0,y0,x1,y1 ...]
#                                               # redo only the track from an existing map (after looking at the
#                                               #   _check.png: --block areas that are not part of the course)
# NAME: letters and digits only (no _), e.g. Practice1126. The run name is the bag's (run_MMDD_HHMM).
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "$PY" ]; then                     # override: PY=/path/to/python tools/map_session.sh ...
  PY=python3
  for p in "$HOME/venv/torch210/bin/python" "$HOME/venv/f1tenth/bin/python"; do [ -x "$p" ] && PY="$p" && break; done
fi

track() {  # map dir, name, extra args for map_to_track.py
  local MAP="$1" NAME="$2"; shift 2
  "$PY" "$REPO/tools/map_to_track.py" "$MAP" --name "$NAME" "$@"
  # plan B: the preview model drives on this course next (MODEL=preview tools/drive_session.sh). Written before the
  # simulator check, so a failing check (set -e) does not leave the old course in the config (11/25 review)
  local CFG RUN
  RUN="$(basename "${MAP%/}")"
  for CFG in "$REPO"/src/f1tenth_ros/f1tenth_stack/config/agent_preview*.yaml; do    # fast (IL 13) and robust model
    sed -i -e "s|^\(    course:\).*|\1 src/f1tenth/installation/maps_real/$NAME.npz|" \
           -e "s|^\(    map_dir:\).*|\1 maps/$RUN|" "$CFG"
    if grep -q "^    course: src/f1tenth/installation/maps_real/$NAME.npz$" "$CFG" && grep -q "^    map_dir: maps/$RUN$" "$CFG"; then
      echo "[preview] $(basename "$CFG") -> course $NAME.npz, map maps/$RUN"
    else
      echo "[error] could not write course / map_dir into $CFG: set them by hand"
    fi
  done
  "$PY" "$REPO/tools/check_track.py" "$REPO/src/f1tenth/installation/maps_real/$NAME.npz" --models ppo_s2r/12190 --caps 8 4 2
  echo "[next] look at maps_real/${NAME}_check.png and ${NAME}_drive.png; if a room / opening is part of the corridor:"
  echo "       tools/map_session.sh --rebuild $MAP $NAME --block x0,y0,x1,y1   (coordinates from the png)"
  echo "       then commit: git add -f maps/$(basename "$MAP") src/f1tenth/installation/maps_real/$NAME* src/f1tenth_ros/f1tenth_stack/config/agent_preview*.yaml && git commit && git push"
}

case "$1" in
  --finish)
    NAME="$2"; [ -n "$NAME" ] || { echo "usage: $0 --finish NAME [--t0 S --t1 S]"; exit 1; }
    shift 2
    "$REPO/tools/drive_session.sh" --stop
    OUT=$("$REPO/tools/drive_session.sh" --fetch | tee /dev/stderr | sed -n 's/^\[done\] //p')
    [ -n "$OUT" ] || { echo "[error] fetch failed"; exit 1; }
    "$PY" "$REPO/tools/eval_drive.py" "$OUT" | tee "$OUT/eval.txt" || true
    MAP="$REPO/maps/$(basename "$OUT" _extracted)"
    "$PY" "$REPO/tools/map_offline.py" "$OUT" "$MAP" "$@"
    track "$MAP" "$NAME"
    exit 0 ;;
  --rebuild)
    MAP="$2" NAME="$3"; [ -n "$MAP" ] && [ -n "$NAME" ] || { echo "usage: $0 --rebuild maps/<run> NAME [--block ...]"; exit 1; }
    shift 3
    track "$MAP" "$NAME" "$@"
    exit 0 ;;
esac

MODEL=mapping exec "$REPO/tools/drive_session.sh" "$@"
