#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/ros2_ws"

set +u
source /opt/ros/humble/setup.bash
set -u
colcon build --symlink-install

echo "Built EdgeAI-ROS workspace."
echo "Run: source $ROOT/ros2_ws/install/setup.bash"
