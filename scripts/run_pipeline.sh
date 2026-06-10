#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${1:-predictive_adaptive}"
DEADLINE_MS="${2:-33.0}"
IMAGE_DIR="${IMAGE_DIR:-$ROOT/data/test_images}"
CSV_PATH="${CSV_PATH:-$ROOT/data/results/edgeai_ros_${POLICY}.csv}"
MODELS="${MODELS:-repvit_m0_9,repvit_m0_9,repvit_m0_9}"
IMAGE_SIZES="${IMAGE_SIZES:-160,192,224}"
QUALITY_SCORES="${QUALITY_SCORES:-0.72,0.86,1.0}"
DEVICE="${DEVICE:-auto}"
DEVICES="${DEVICES:-none}"
CACHE_SIZE="${CACHE_SIZE:-2}"
PRELOAD_MODELS="${PRELOAD_MODELS:-false}"
PERIOD_SEC="${PERIOD_SEC:-0.5}"
QOS_DEPTH="${QOS_DEPTH:-10}"
QOS_RELIABILITY="${QOS_RELIABILITY:-reliable}"
QUEUE_POLICY="${QUEUE_POLICY:-fifo}"
CPU_AFFINITY="${CPU_AFFINITY:-none}"
NICE_DELTA="${NICE_DELTA:-0}"
DROP_STALE_MS="${DROP_STALE_MS:-0.0}"
OVERLOAD_RHO="${OVERLOAD_RHO:-1.0}"
STARTUP_DELAY_SEC="${STARTUP_DELAY_SEC:-0}"

set +u
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
set -u

mkdir -p "$ROOT/data/results"

echo "Starting EdgeAI-ROS pipeline with policy=$POLICY deadline_ms=$DEADLINE_MS"
echo "This script starts three ROS nodes in the foreground via background jobs."

ros2 run edgeai_ros edgeai_logger \
  --ros-args \
  -p csv_path:="$CSV_PATH" \
  -p qos_depth:="$QOS_DEPTH" \
  -p qos_reliability:="$QOS_RELIABILITY" &
LOGGER_PID=$!

ros2 run edgeai_ros adaptive_repvit_node \
  --ros-args \
  -p policy:="$POLICY" \
  -p models:="$MODELS" \
  -p image_sizes:="$IMAGE_SIZES" \
  -p quality_scores:="$QUALITY_SCORES" \
  -p devices:="$DEVICES" \
  -p deadline_ms:="$DEADLINE_MS" \
  -p device:="$DEVICE" \
  -p cache_size:="$CACHE_SIZE" \
  -p preload_models:="$PRELOAD_MODELS" \
  -p qos_depth:="$QOS_DEPTH" \
  -p qos_reliability:="$QOS_RELIABILITY" \
  -p queue_policy:="$QUEUE_POLICY" \
  -p cpu_affinity:="$CPU_AFFINITY" \
  -p nice_delta:="$NICE_DELTA" \
  -p drop_stale_ms:="$DROP_STALE_MS" \
  -p overload_rho:="$OVERLOAD_RHO" \
  -p checkpoint_dir:="$ROOT/checkpoints" \
  -p repvit_path:="${REPVIT_PATH:-/workspace/OS2026/RepViT}" &
INFER_PID=$!

if awk "BEGIN {exit !($STARTUP_DELAY_SEC > 0)}"; then
  echo "Waiting ${STARTUP_DELAY_SEC}s before starting image publisher..."
  sleep "$STARTUP_DELAY_SEC"
fi

ros2 run edgeai_ros image_publisher \
  --ros-args \
  -p image_dir:="$IMAGE_DIR" \
  -p period_sec:="$PERIOD_SEC" \
  -p qos_depth:="$QOS_DEPTH" \
  -p qos_reliability:="$QOS_RELIABILITY" &
PUB_PID=$!

trap 'kill $PUB_PID $INFER_PID $LOGGER_PID 2>/dev/null || true' EXIT
wait
