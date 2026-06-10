#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STRESS_MODE="${1:-cpu}"
POLICY="${2:-predictive_adaptive}"
MODE="${3:-vit}"
DEADLINE_MS="${4:-33.0}"

set +u
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
set -u

ros2 run edgeai_ros stress_node --ros-args -p mode:="$STRESS_MODE" &
STRESS_PID=$!
trap 'kill $STRESS_PID 2>/dev/null || true' EXIT

if [[ "$MODE" == "concurrent" ]]; then
  "$ROOT/scripts/run_concurrent_pipeline.sh" "$POLICY" "$DEADLINE_MS"
else
  "$ROOT/scripts/run_pipeline.sh" "$POLICY" "$DEADLINE_MS"
fi
