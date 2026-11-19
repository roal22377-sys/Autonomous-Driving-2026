#!/bin/bash
# Steering servo range check ON THE CAR, wheels in the air: bringup + tools/servo_range_check.py.
set -e
cd ~/ros2_ws
set +u
source /opt/ros/foxy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-6}"

if pgrep -f "[r]os2 launch f1tenth_stack|[a]gent_node" >/dev/null; then
  echo "[error] bringup/agent already running. Stop it first."; exit 1
fi
read -p "바퀴를 띄웠나요 (상자 위)? Enter로 시작, Ctrl+C로 취소: "

cleanup() { pkill -INT -f "[r]os2 launch f1tenth_stack" 2>/dev/null || true; sleep 2; }
trap cleanup EXIT
echo "[bringup] (log: /tmp/servo_bringup.log)"
ros2 launch f1tenth_stack bringup_launch.py \
  vesc_config:=$HOME/ros2_ws/src/f1tenth_ros/f1tenth_stack/config/vesc.yaml use_agent:=false \
  > /tmp/servo_bringup.log 2>&1 &
sleep 6
python3 tools/servo_range_check.py
