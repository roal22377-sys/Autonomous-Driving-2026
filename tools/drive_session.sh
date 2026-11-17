#!/bin/bash
# One command on the local PC for an autonomous-driving session on the car.
#   tools/drive_session.sh            # open 5 tabs: bringup / teleop(e-stop) / bag record / cmd monitor / agent
#   tools/drive_session.sh --stop     # stop agent, bag (saved cleanly), bringup on the car
#   tools/drive_session.sh --fetch    # extract the newest bag on the car -> logs/<bag>_extracted/
# The agent tab waits for Enter, so press 's' in the teleop tab first (holds the car stopped);
# then 'r' = drive, 's' = stop.
set -e
CAR_HOST="${CAR_HOST:-jowonjun@10.106.0.73}"
SSH_OPTS="-o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3"
CAR_ENV='source ~/venv/torch210/bin/activate && cd ~/ros2_ws && source /opt/ros/foxy/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=6'
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! ssh $SSH_OPTS -o BatchMode=yes "$CAR_HOST" true; then
  echo "[error] cannot reach $CAR_HOST (car off / not on Wi-Fi?)"; exit 1
fi

case "$1" in
  --stop)
    # SIGINT so ros2 bag closes the db cleanly; agent first so nothing drives while bringup goes down
    ssh $SSH_OPTS "$CAR_HOST" 'pkill -INT -f "[a]gent_node"; sleep 1; pkill -INT -f "[r]os2 bag record"; sleep 2;
      pkill -INT -f "[r]os2 launch f1tenth_stack"; pkill -INT -f "[k]ey_teleop"; pkill -INT -f "[t]opic echo"; sleep 2;
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
  "") ;;
  *) echo "usage: $0 [--stop|--fetch]"; exit 1 ;;
esac

if ssh $SSH_OPTS "$CAR_HOST" 'pgrep -f "[r]os2 launch f1tenth_stack|[a]gent_node" >/dev/null'; then
  echo "[error] bringup/agent already running on the car. Run: $0 --stop"; exit 1
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
mktab monitor  "sleep 6"          'ros2 topic echo /ackermann_cmd'
mktab agent    "echo 'Wait for bringup, press s in the TELEOP tab, put the car on the track,'; read -p 'then press Enter here to start the agent: '" \
               'ros2 run f1tenth_stack agent_node --ros-args --params-file src/f1tenth_ros/f1tenth_stack/config/agent.yaml'

gnome-terminal --window --title "1 bringup"            -e "bash $TABS/bringup.sh" \
               --tab    --title "2 TELEOP s=stop r=go" -e "bash $TABS/teleop.sh" \
               --tab    --title "3 bag record"         -e "bash $TABS/bag.sh" \
               --tab    --title "4 cmd monitor"        -e "bash $TABS/monitor.sh" \
               --tab    --title "5 agent"              -e "bash $TABS/agent.sh" 2>/dev/null

echo "[local] opened 5 tabs. Stop everything with: $0 --stop ; fetch the log with: $0 --fetch"
