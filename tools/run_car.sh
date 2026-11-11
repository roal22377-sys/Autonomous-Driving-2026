#!/bin/bash
# Pull the latest code/params from GitHub, then start the car bringup
# using vesc.yaml straight from src/ (no colcon build needed for param changes).
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

git pull --ff-only

if [ -f "$REPO/install/setup.bash" ]; then
  source "$REPO/install/setup.bash"
fi

exec ros2 launch f1tenth_stack bringup_launch.py \
  vesc_config:="$REPO/src/f1tenth_ros/f1tenth_stack/config/vesc.yaml" "$@"
