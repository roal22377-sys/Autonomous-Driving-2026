#!/bin/bash
# Run on the local PC: push to GitHub, make the car git pull over SSH, optionally start bringup there.
#   tools/deploy_car.sh                      # push, car pulls
#   tools/deploy_car.sh -m "message"         # commit all changes first
#   tools/deploy_car.sh --launch             # ...then start bringup (use_agent:=false) on the car; Ctrl+C stops it
#   tools/deploy_car.sh --launch-agent       # ...then start bringup with the policy (agent_node)
#   tools/deploy_car.sh --build              # also colcon build f1tenth_stack (new files / setup.py changes)
# yaml params are read straight from src/ (vesc_config:=...), so param changes need no build.
# Override target with CAR_HOST / CAR_WS / CAR_DOMAIN_ID env vars.
set -e
CAR_HOST="${CAR_HOST:-jowonjun@10.106.0.73}"
CAR_WS="${CAR_WS:-\$HOME/ros2_ws}"
CAR_DOMAIN_ID="${CAR_DOMAIN_ID:-6}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# fail fast instead of hanging when the car's Wi-Fi drops (no answer for ~15 s -> ssh exits)
SSH_OPTS=(-o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)

MSG="" BUILD=0 LAUNCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    -m) [ -n "$2" ] || { echo "usage: $0 -m \"commit message\""; exit 1; }
        MSG="$2"; shift 2 ;;
    --build) BUILD=1; shift ;;
    --launch) LAUNCH="use_agent:=false"; shift ;;
    --launch-agent) LAUNCH="use_agent:=true"; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

if [ -n "$MSG" ]; then
  git add -A
  git diff --cached --quiet || git commit -m "$MSG"
elif [ -n "$(git status --porcelain)" ]; then
  echo "[warn] uncommitted changes are NOT deployed (use -m \"msg\" to commit them):"
  git status --short
fi

echo "[local] git push"
git push

echo "[car] git pull on $CAR_HOST"
if ! ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$CAR_HOST" true; then
  echo "[error] cannot reach $CAR_HOST (car off / not on Wi-Fi?). GitHub is updated; rerun when the car is up."
  exit 1
fi
ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$CAR_HOST" "set -e
  cd $CAR_WS && timeout 60 git pull --ff-only && git log --oneline -1
  if [ $BUILD = 1 ]; then
    source /opt/ros/foxy/setup.bash
    cd $CAR_WS && colcon build --symlink-install --packages-select f1tenth_stack
  fi"

if [ -n "$LAUNCH" ]; then
  echo "[car] bringup ($LAUNCH) -- Ctrl+C to stop"
  exec ssh "${SSH_OPTS[@]}" -t "$CAR_HOST" "cd $CAR_WS && export PYTHONPATH=\$HOME/venv/torch210/lib/python3.8/site-packages:\$PYTHONPATH && source \$HOME/venv/torch210/bin/activate && source /opt/ros/foxy/setup.bash && source $CAR_WS/install/setup.bash && \
    export ROS_DOMAIN_ID=$CAR_DOMAIN_ID && \
    ros2 launch f1tenth_stack bringup_launch.py \
      vesc_config:=$CAR_WS/src/f1tenth_ros/f1tenth_stack/config/vesc.yaml $LAUNCH"
fi
echo "[done] car updated to $(git log --oneline -1). Restart bringup to apply."
