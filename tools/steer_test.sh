#!/bin/bash
# Steering delay measurement in one command, ON THE CAR (or: ssh -t <car> '~/ros2_ws/tools/steer_test.sh').
#   tools/steer_test.sh                         # default: 1.5 m/s, +-0.3 rad, 0.6 s holds, 2 cycles
#   tools/steer_test.sh --speed 2.0 --cycles 1  # any tools/steer_step_test.py option
#   tools/steer_test.sh --stand                 # wheels in the air: servo delay only, read from slow-mo video
# 1) bringup (use_agent:=false)  2) bag record  3) scripted steering steps (Enter to start, Ctrl+C stops)
# 4) stop bag + bringup  5) extract the bag  6) tools/steer_delay.py on it
set -e
cd ~/ros2_ws
set +u
source /opt/ros/foxy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-6}"

if pgrep -f "[r]os2 launch f1tenth_stack|[a]gent_node" >/dev/null; then
  echo "[error] bringup/agent already running. Stop it first (tools/drive_session.sh --stop on the PC)."; exit 1
fi

STAND=0
for arg in "$@"; do [ "$arg" = "--stand" ] && STAND=1; done

python3 tools/steer_step_test.py --dry_run "$@"
if [ $STAND = 1 ]; then
  read -p "Wheels in the air (car on a box)? Enter to continue, Ctrl+C to cancel: "
else
  read -p "Enough space for that? Enter to continue, Ctrl+C to cancel: "
fi

BAG=~/bags/steer_$(date +%m%d_%H%M)
mkdir -p ~/bags
cleanup() {
  echo "[stop] bag + bringup"
  [ -n "$BAG_PID" ] && kill -INT "$BAG_PID" 2>/dev/null && sleep 2
  pkill -INT -f "[r]os2 launch f1tenth_stack" 2>/dev/null || true
  sleep 2
}
trap cleanup EXIT

echo "[1/4] bringup (log: /tmp/steer_bringup.log)"
ros2 launch f1tenth_stack bringup_launch.py \
  vesc_config:=$HOME/ros2_ws/src/f1tenth_ros/f1tenth_stack/config/vesc.yaml use_agent:=false \
  > /tmp/steer_bringup.log 2>&1 &
sleep 6

echo "[2/4] recording $BAG"
ros2 bag record -o "$BAG" /scan /odom /drive /teleop /ackermann_cmd /sensors/core /sensors/servo_position_command \
  > /tmp/steer_bag.log 2>&1 &
BAG_PID=$!
sleep 2

echo "[3/4] step test"
python3 tools/steer_step_test.py "$@" || true
sleep 1

cleanup
trap - EXIT

if [ $STAND = 1 ]; then
  echo "[done] stand test: read the delay from the slow-motion video (bag kept at $BAG)"
  exit 0
fi

echo "[4/4] extract + analyze"
python3 tools/bag_extract.py "$BAG"
python3 tools/steer_delay.py "${BAG}_extracted" | tee "${BAG}_extracted/steer_delay.txt"
echo "[done] log: ${BAG}_extracted  (result also in steer_delay.txt)"
