#!/bin/bash
# One command on the local PC for an autonomous-driving session on the car.
#   tools/drive_session.sh            # git push + car git pull (tools/deploy_car.sh), then open 6 tabs:
#                                     #   bringup / teleop(e-stop) / bag record / live status / cmd monitor / agent
#   tools/drive_session.sh -m "msg"   # ...commit all local changes first, then push/pull
#   tools/drive_session.sh --no-deploy  # skip the push/pull, just open the tabs
#   tools/drive_session.sh --finish   # stop everything, fetch the newest bag, show tools/eval_drive.py results
#   tools/drive_session.sh --circle   # same tabs, but the last one runs the circle tests (tools/circle_series.sh)
#                                     #   instead of the agent: steering map, tire grip, true speed, low-speed hold
#   tools/drive_session.sh --finish --circle   # ...then stop + fetch + tools/circle_analysis.py
#   tools/drive_session.sh --stop     # stop agent, bag (saved cleanly), bringup on the car
#   tools/drive_session.sh --fetch    # extract the newest bag on the car -> logs/<bag>_extracted/
#   tools/drive_session.sh --agent    # (re)start only the agent in this terminal (bringup keeps running)
# ros2 run uses /usr/bin/python3, so torch comes from PYTHONPATH (venv site-packages), not from activate.
# The agent tab waits for Enter, so press 's' in the teleop tab first (holds the car stopped);
# then 'r' = drive, 's' = stop.
set -e
CAR_HOST="${CAR_HOST:-jowonjun@10.106.0.73}"
SSH_OPTS="-o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3"
CAR_ENV='export PYTHONPATH=$HOME/venv/torch210/lib/python3.8/site-packages:$PYTHONPATH && source ~/venv/torch210/bin/activate && cd ~/ros2_ws && source /opt/ros/foxy/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=6'
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! ssh $SSH_OPTS -o BatchMode=yes "$CAR_HOST" true; then
  echo "[error] cannot reach $CAR_HOST (car off / not on Wi-Fi?)"; exit 1
fi

DEPLOY=1 MSG="" CIRCLE=0
case "$1" in
  --no-deploy) DEPLOY=0; shift ;;
  -m) [ -n "$2" ] || { echo "usage: $0 -m \"commit message\""; exit 1; }
      MSG="$2"; shift 2 ;;
esac
if [ "$1" = "--circle" ]; then CIRCLE=1; shift; fi

case "$1" in
  --finish)
    "$0" --stop
    OUT=$("$0" --fetch | tee /dev/stderr | sed -n 's/^\[done\] //p')
    [ -n "$OUT" ] || { echo "[error] fetch failed"; exit 1; }
    PY=python3; [ -x "$HOME/venv/torch210/bin/python" ] && PY="$HOME/venv/torch210/bin/python"
    if [ "$2" = "--circle" ]; then
      "$PY" "$REPO/tools/circle_analysis.py" "$OUT" | tee "$OUT/circle.txt"
      echo "[done] analysis saved to $OUT/circle.txt"
    else
      "$PY" "$REPO/tools/eval_drive.py" "$OUT" | tee "$OUT/eval.txt"
      echo "[done] analysis saved to $OUT/eval.txt"
    fi
    exit 0 ;;
  --stop)
    # SIGINT so ros2 bag closes the db cleanly; agent first so nothing drives while bringup goes down
    ssh $SSH_OPTS "$CAR_HOST" 'pkill -INT -f "[a]gent_node"; pkill -INT -f "[c]ircle_test.py"; pkill -f "[c]ircle_series.sh"; sleep 1; pkill -INT -f "[r]os2 bag record"; sleep 2;
      pkill -INT -f "[r]os2 launch f1tenth_stack"; pkill -INT -f "[k]ey_teleop"; pkill -INT -f "[t]opic echo";
      pkill -INT -f "[l]ive_status.py"; sleep 2;
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
  --agent)
    # (re)start only the agent in this terminal, e.g. after it crashed; bringup keeps running
    if ssh $SSH_OPTS "$CAR_HOST" 'pgrep -f "[a]gent_node" >/dev/null'; then
      echo "[error] agent_node already running on the car"; exit 1
    fi
    read -p "Press s in the TELEOP tab first, then Enter here to start the agent: "
    exec ssh $SSH_OPTS -t "$CAR_HOST" "$CAR_ENV && ros2 run f1tenth_stack agent_node --ros-args --params-file src/f1tenth_ros/f1tenth_stack/config/agent.yaml" ;;
  "") ;;
  *) echo "usage: $0 [-m msg|--no-deploy] [--circle] | --finish [--circle] | --stop | --fetch | --agent"; exit 1 ;;
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
mktab bag      "sleep 6"          'mkdir -p ~/bags && ros2 bag record -o ~/bags/run_$(date +%m%d_%H%M) /scan /odom /drive /teleop /ackermann_cmd /sensors/core'
mktab status   "sleep 6"          'python3 tools/live_status.py'
mktab monitor  "sleep 6"          'ros2 topic echo /ackermann_cmd'
if [ $CIRCLE = 1 ]; then
  LAST="6 CIRCLE test"
  mktab agent  "echo 'Circle tests: wait for bringup, press r in the TELEOP tab (s = emergency stop).'; sleep 4" \
               'tools/circle_series.sh'
else
  LAST="6 agent"
  mktab agent  "echo 'Wait for bringup, press s in the TELEOP tab, put the car on the track,'; read -p 'then press Enter here to start the agent: '" \
               'ros2 run f1tenth_stack agent_node --ros-args --params-file src/f1tenth_ros/f1tenth_stack/config/agent.yaml'
fi

gnome-terminal --window --title "1 bringup"            -e "bash $TABS/bringup.sh" \
               --tab    --title "2 TELEOP s=stop r=go" -e "bash $TABS/teleop.sh" \
               --tab    --title "3 bag record"         -e "bash $TABS/bag.sh" \
               --tab    --title "4 LIVE status"        -e "bash $TABS/status.sh" \
               --tab    --title "5 cmd monitor"        -e "bash $TABS/monitor.sh" \
               --tab    --title "$LAST"                -e "bash $TABS/agent.sh" 2>/dev/null

if [ $CIRCLE = 1 ]; then
  echo "[local] opened 6 tabs (circle test). When done: $0 --finish --circle"
else
  echo "[local] opened 6 tabs. When done: $0 --finish  (stop + fetch the log + analysis)"
fi
