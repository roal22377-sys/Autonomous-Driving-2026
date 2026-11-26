#!/bin/bash
# One command on the local PC for an autonomous-driving session on the car.
#   tools/drive_session.sh            # git push + car git pull (tools/deploy_car.sh), then open 6 tabs:
#                                     #   bringup / teleop(e-stop) / bag record / live status / cmd monitor / agent
#   tools/drive_session.sh -m "msg"   # ...commit all local changes first, then push/pull
#   tools/drive_session.sh --no-deploy  # skip the push/pull, just open the tabs
#   tools/drive_session.sh --finish   # stop everything, fetch the newest bag, show tools/eval_drive.py results
#   tools/drive_session.sh --circle   # same tabs, but the last one runs the circle tests (tools/circle_series.sh)
#                                     #   instead of the agent: steering map, tire grip, true speed, low-speed hold
#   tools/drive_session.sh --circle radius      # full lock left/right, slow circles (turning radius, tape)
#   tools/drive_session.sh --finish --circle   # ...then stop + fetch + tools/circle_analysis.py
#   tools/drive_session.sh --speed [2 3 4]      # last tab runs straight speed tests toward a wall (tools/speed_series.sh)
#   tools/drive_session.sh --finish --speed    # ...then stop + fetch + tools/speed_analysis.py (speed_to_erpm_gain)
#   tools/drive_session.sh --straight [--trim -0.01]  # last tab runs the straight-line alignment (tools/straight_test.py):
#                                     #   drive straight, enter the drift from a tape line, it suggests the steering trim
#   tools/drive_session.sh --stop     # stop agent, bag (saved cleanly), bringup on the car
#   tools/drive_session.sh --fetch    # extract the newest bag on the car -> logs/<bag>_extracted/
#   tools/drive_session.sh --agent    # (re)start only the agent in this terminal (bringup keeps running)
#   tools/drive_session.sh --fetch-all run_1126   # every bag of that day not in logs/ yet (extract on the car + copy)
#   MODEL=preview tools/drive_session.sh --loc  # live localization check (OK/CHECK/WRONG/LOST + picture), 2nd terminal
#   MODEL=il tools/drive_session.sh   # IL only (config/agent_il.yaml);  MODEL=ilppo: IL + PPO (agent_ilppo.yaml)
#   MODEL=preview tools/drive_session.sh  # plan B model on the last mapped course (agent_preview.yaml, MAP_TOOL.md)
#                                     # without MODEL: config/agent.yaml as before
# ros2 run uses /usr/bin/python3, so torch comes from PYTHONPATH (venv site-packages), not from activate.
# The agent tab waits for Enter, so press 's' in the teleop tab first (holds the car stopped);
# then 'r' = drive, 's' = stop.
set -e
AGENT_YAML="agent${MODEL:+_$MODEL}.yaml"
CAR_HOST="${CAR_HOST:-jowonjun@10.106.0.73}"
# 11/24: Wi-Fi round trips reached 0.2 s and 5 s connects failed now and then -> 10 s and up to 3 tries
SSH_OPTS="-o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3"
CAR_ENV='export PYTHONPATH=$HOME/venv/torch210/lib/python3.8/site-packages:$PYTHONPATH && source ~/venv/torch210/bin/activate && cd ~/ros2_ws && source /opt/ros/foxy/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=6'
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REACHED=0
for TRY in 1 2 3; do
  if ssh $SSH_OPTS -o BatchMode=yes "$CAR_HOST" true 2>/dev/null; then REACHED=1; break; fi
  echo "[warn] no answer from $CAR_HOST (try $TRY/3)"; sleep 2
done
[ $REACHED = 1 ] || { echo "[error] cannot reach $CAR_HOST (car off / not on Wi-Fi?)"; exit 1; }

DEPLOY=1 MSG="" CIRCLE=0
case "$1" in
  --no-deploy) DEPLOY=0; shift ;;
  -m) [ -n "$2" ] || { echo "usage: $0 -m \"commit message\""; exit 1; }
      MSG="$2"; shift 2 ;;
esac
CIRCLE_ARGS="" SPEED=0 SPEED_ARGS="" STRAIGHT=0 STRAIGHT_ARGS=""
if [ "$1" = "--straight" ]; then
  STRAIGHT=1; shift
  while [[ "$1" == --* && "$1" != --finish && "$1" != --stop ]]; do STRAIGHT_ARGS="$STRAIGHT_ARGS $1 $2"; shift 2; done
fi
if [ "$1" = "--speed" ]; then
  SPEED=1; shift
  while [[ "$1" =~ ^[0-9.]+$ ]]; do SPEED_ARGS="$SPEED_ARGS $1"; shift; done
fi
if [ "$1" = "--circle" ]; then
  CIRCLE=1; shift
  # --circle radius: full lock left + right, slow steady circles for measuring the turning radius
  if [ "$1" = "radius" ]; then CIRCLE_ARGS="radius"; shift; fi
fi

case "$1" in
  --finish)
    "$0" --stop
    OUT=$("$0" --fetch | tee /dev/stderr | sed -n 's/^\[done\] //p')
    [ -n "$OUT" ] || { echo "[error] fetch failed"; exit 1; }
    # PC python with numpy/pandas/matplotlib (+ torch for the sim checks); override: PY=/path/to/python
    if [ -z "$PY" ]; then
      PY=python3
      for p in "$HOME/venv/torch210/bin/python" "$HOME/venv/f1tenth/bin/python"; do [ -x "$p" ] && PY="$p" && break; done
    fi
    if [ "$2" = "--circle" ]; then
      "$PY" "$REPO/tools/circle_analysis.py" "$OUT" | tee "$OUT/circle.txt"
      echo "[done] analysis saved to $OUT/circle.txt"
    elif [ "$2" = "--speed" ]; then
      "$PY" "$REPO/tools/speed_analysis.py" "$OUT" | tee "$OUT/speed.txt"
      echo "[done] analysis saved to $OUT/speed.txt"
    else
      "$PY" "$REPO/tools/eval_drive.py" "$OUT" | tee "$OUT/eval.txt"
      echo "[done] analysis saved to $OUT/eval.txt"
    fi
    exit 0 ;;
  --stop)
    # SIGINT so ros2 bag closes the db cleanly; agent first so nothing drives while bringup goes down
    ssh $SSH_OPTS "$CAR_HOST" 'pkill -INT -f "[a]gent_node"; pkill -INT -f "[c]ircle_test.py"; pkill -f "[c]ircle_series.sh"; pkill -INT -f "[s]peed_test.py"; pkill -f "[s]peed_series.sh"; sleep 1; pkill -INT -f "[r]os2 bag record"; sleep 2;
      pkill -INT -f "[r]os2 launch f1tenth_stack"; pkill -INT -f "[k]ey_teleop"; pkill -INT -f "[t]opic echo";
      pkill -INT -f "[l]ive_status.py"; sleep 2;
      pkill -INT -f "[k]eyboard_to_ackermann"; pkill -INT -f "[r]os2_ws/install/"; pkill -INT -f "[u]rg_node"; sleep 1;
      pkill -f "[r]os2_ws/install/"; pkill -f "[u]rg_node";
      pgrep -af "[r]os2|[a]gent_node|[u]rg_node|[v]esc_" || echo "[car] all stopped"'
    exit 0 ;;
  --fetch)
    BAG=$(ssh $SSH_OPTS "$CAR_HOST" 'ls -td ~/bags/run_* 2>/dev/null | grep -v _extracted | head -1')
    [ -n "$BAG" ] || { echo "[error] no bag in ~/bags on the car"; exit 1; }
    echo "[car] extracting $BAG"
    ssh $SSH_OPTS "$CAR_HOST" "$CAR_ENV && python3 tools/bag_extract.py $BAG"
    mkdir -p "$REPO/logs"
    scp -rq "$CAR_HOST:${BAG}_extracted" "$REPO/logs/"
    echo "[done] $REPO/logs/$(basename "$BAG")_extracted"
    exit 0 ;;
  --fetch-all)
    # every bag on the car that is not in logs/ yet (e.g. sessions ended without --finish, car switched off):
    # extract on the car, copy here. Optional: a date prefix, e.g. --fetch-all run_1126
    PREFIX="${2:-run_}"
    mkdir -p "$REPO/logs"
    BAGS=$(ssh $SSH_OPTS "$CAR_HOST" "ls -d ~/bags/${PREFIX}* 2>/dev/null" | grep -v _extracted)
    [ -n "$BAGS" ] || { echo "[error] no bag ~/bags/${PREFIX}* on the car"; exit 1; }
    for BAG in $BAGS; do
      N=$(basename "$BAG")
      if [ -d "$REPO/logs/${N}_extracted" ]; then echo "[skip] $N (already in logs/)"; continue; fi
      echo "[car] extracting $N"
      if ssh $SSH_OPTS "$CAR_HOST" "$CAR_ENV && python3 tools/bag_extract.py $BAG" && scp -rq "$CAR_HOST:${BAG}_extracted" "$REPO/logs/"; then
        echo "[done] logs/${N}_extracted"
      else
        echo "[error] $N failed (bag still on the car)"
      fi
    done
    echo "[next] git add -f logs/${PREFIX}*_extracted && git commit -m 'car logs' && git push"
    exit 0 ;;
  --agent)
    # (re)start only the agent in this terminal, e.g. after it crashed; bringup keeps running
    if ssh $SSH_OPTS "$CAR_HOST" 'pgrep -f "[a]gent_node" >/dev/null'; then
      echo "[error] agent_node already running on the car"; exit 1
    fi
    read -p "Press s in the TELEOP tab first, then Enter here to start the agent: "
    exec ssh $SSH_OPTS -t "$CAR_HOST" "$CAR_ENV && ros2 run f1tenth_stack agent_node --ros-args --params-file src/f1tenth_ros/f1tenth_stack/config/$AGENT_YAML" ;;
  --loc)
    # live localization check of the plan-B models in this terminal (read-only; tools/loc_view.py): OK / CHECK /
    # WRONG / LOST + the map around the car with the scan on it. Run in a second PC terminal next to the session.
    exec ssh $SSH_OPTS -t "$CAR_HOST" "$CAR_ENV && python3 tools/loc_view.py --yaml src/f1tenth_ros/f1tenth_stack/config/$AGENT_YAML" ;;
  "") ;;
  *) echo "usage: $0 [-m msg|--no-deploy] [--circle [radius] | --speed [v ...] | --straight [--trim x]] | --finish [--circle|--speed] | --stop | --fetch | --fetch-all [run_MMDD] | --agent | --loc"; exit 1 ;;
esac


if ssh $SSH_OPTS "$CAR_HOST" 'pgrep -f "[r]os2 launch f1tenth_stack|[a]gent_node" >/dev/null'; then
  echo "[error] bringup/agent already running on the car. Run: $0 --stop"; exit 1
fi

if [ $DEPLOY = 1 ]; then
  # push local commits and make the car pull, so the car drives exactly what is in the repo
  if [ -n "$MSG" ]; then "$REPO/tools/deploy_car.sh" -m "$MSG"; else "$REPO/tools/deploy_car.sh"; fi
fi

# each tab runs a small script: ssh to the car with the env set up, run one command, keep the tab open
TABS="$(mktemp -d /tmp/drive_session.XXXX)"
mktab() {  # name, local pre-step, remote command
  cat > "$TABS/$1.sh" <<EOF
$2
ssh $SSH_OPTS -t $CAR_HOST '$CAR_ENV && $3'
echo; read -p "[ended] Enter to close this tab"
EOF
}
mktab bringup  ""                 'ros2 launch f1tenth_stack bringup_launch.py vesc_config:=$HOME/ros2_ws/src/f1tenth_ros/f1tenth_stack/config/vesc.yaml use_agent:=false'
mktab teleop   "sleep 4"          'ros2 run key_teleop key_teleop'
# a session restarted within the same minute would reuse the folder name and ros2 bag record would refuse -> add seconds then
mktab bag      "sleep 6"          'mkdir -p ~/bags && N=run_$(date +%m%d_%H%M) && { [ ! -e ~/bags/$N ] || N=${N}_$(date +%S); } && ros2 bag record -o ~/bags/$N /scan /odom /drive /teleop /ackermann_cmd /sensors/core /loc_pose'
mktab status   "sleep 6"          'python3 tools/live_status.py'
mktab monitor  "sleep 6"          'ros2 topic echo /ackermann_cmd'
if [ $STRAIGHT = 1 ]; then
  LAST="6 STRAIGHT test"
  mktab agent  "echo 'Straight alignment: wait for bringup, press r in the TELEOP tab (s = emergency stop).'; sleep 6" \
               "python3 tools/straight_test.py $STRAIGHT_ARGS"
elif [ $SPEED = 1 ]; then
  LAST="6 SPEED test"
  mktab agent  "echo 'Speed tests: wait for bringup, press r in the TELEOP tab (s = emergency stop).'; sleep 4" \
               "tools/speed_series.sh $SPEED_ARGS"
elif [ $CIRCLE = 1 ]; then
  LAST="6 CIRCLE test"
  mktab agent  "echo 'Circle tests: wait for bringup, press r in the TELEOP tab (s = emergency stop).'; sleep 4" \
               "tools/circle_series.sh $CIRCLE_ARGS"
else
  LAST="6 agent"
  mktab agent  "echo 'Wait for bringup, press s in the TELEOP tab, put the car on the track,'; read -p 'then press Enter here to start the agent: '" \
               "ros2 run f1tenth_stack agent_node --ros-args --params-file src/f1tenth_ros/f1tenth_stack/config/$AGENT_YAML"
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
  # another computer (Mac, WSL without a desktop, ...): open 6 terminals yourself and run one line in each, in order
  echo "[local] gnome-terminal not found. Open 6 terminals and run, in this order (wait for bringup before the rest):"
  echo "  1 bringup:  bash $TABS/bringup.sh"
  echo "  2 TELEOP:   bash $TABS/teleop.sh      (s = stop, r = go)"
  echo "  3 bag:      bash $TABS/bag.sh"
  echo "  4 status:   bash $TABS/status.sh"
  echo "  5 monitor:  bash $TABS/monitor.sh"
  echo "  6 ${LAST#6 }:  bash $TABS/agent.sh"
  exit 0
fi
gnome-terminal --window --title "1 bringup"            -e "bash $TABS/bringup.sh" \
               --tab    --title "2 TELEOP s=stop r=go" -e "bash $TABS/teleop.sh" \
               --tab    --title "3 bag record"         -e "bash $TABS/bag.sh" \
               --tab    --title "4 LIVE status"        -e "bash $TABS/status.sh" \
               --tab    --title "5 cmd monitor"        -e "bash $TABS/monitor.sh" \
               --tab    --title "$LAST"                -e "bash $TABS/agent.sh" 2>/dev/null

if [ $STRAIGHT = 1 ]; then
  echo "[local] opened 6 tabs (straight alignment). When done: $0 --stop, then set steering_angle_to_servo_offset in vesc.yaml"
elif [ $SPEED = 1 ]; then
  echo "[local] opened 6 tabs (speed test). When done: $0 --finish --speed"
elif [ $CIRCLE = 1 ]; then
  echo "[local] opened 6 tabs (circle test). When done: $0 --finish --circle"
else
  echo "[local] opened 6 tabs. When done: $0 --finish  (stop + fetch the log + analysis)"
fi
