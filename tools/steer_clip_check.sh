#!/bin/bash
# On the car: bringup (no agent) + tools/steer_clip_check.py + count vesc_driver "clipping" warnings.
# The motor does not turn (speed forced to 0); only the steering servo moves.
set -e
cd ~/ros2_ws
set +u
export PYTHONPATH=$HOME/venv/torch210/lib/python3.8/site-packages:$PYTHONPATH
source /opt/ros/foxy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-6}"

if pgrep -f "[r]os2 launch f1tenth_stack|[a]gent_node" >/dev/null; then
  echo "[오류] bringup이나 agent가 이미 실행 중이에요. 먼저 끄세요."; exit 1
fi
echo "조향 서보만 좌우 끝까지 움직여요 (모터는 안 돌아요). 바퀴를 띄우면 더 좋아요."
read -p "Enter로 시작, Ctrl+C로 취소: "

LOG=/tmp/clip_bringup.log
cleanup() { pkill -INT -f "[r]os2 launch f1tenth_stack" 2>/dev/null || true; sleep 2; }
trap cleanup EXIT
echo "[bringup] 시작 (로그: $LOG)"
ros2 launch f1tenth_stack bringup_launch.py \
  vesc_config:=$HOME/ros2_ws/src/f1tenth_ros/f1tenth_stack/config/vesc.yaml use_agent:=false > $LOG 2>&1 &
sleep 6
BEFORE=$(grep -c "clipping" $LOG || true)
python3 tools/steer_clip_check.py 2>&1 | grep -v "^\[load\]"
sleep 1
AFTER=$(grep -c "clipping" $LOG || true)
N=$((AFTER - BEFORE))
if [ "$N" -eq 0 ]; then
  echo "[VESC 드라이버] clipping 경고 0회: 서보에서 잘린 명령이 없어요."
else
  echo "[VESC 드라이버] clipping 경고 ${N}회: 아직 잘리는 명령이 있어요. 예:"
  grep "clipping" $LOG | tail -2
fi
