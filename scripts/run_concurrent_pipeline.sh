#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${1:-predictive_adaptive}"
VIT_DEADLINE_MS="${2:-33.0}"
LLM_DEADLINE_MS="${3:-80.0}"
VIT_CSV_PATH="${VIT_CSV_PATH:-$ROOT/data/results/edgeai_ros_vit_${POLICY}.csv}"
LLM_CSV_PATH="${LLM_CSV_PATH:-$ROOT/data/results/edgeai_ros_llm_${POLICY}.csv}"
IMAGE_DIR="${IMAGE_DIR:-$ROOT/data/test_images}"
LLM_MODEL="${LLM_MODEL:-sshleifer/tiny-gpt2}"
LLM_DEVICE="${LLM_DEVICE:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"
PROMPT_PERIOD_SEC="${PROMPT_PERIOD_SEC:-5.0}"
PROMPT_REPEAT="${PROMPT_REPEAT:-1}"
VIT_PERIOD_SEC="${VIT_PERIOD_SEC:-0.5}"
DURATION_SEC="${DURATION_SEC:-0}"
STARTUP_DELAY_SEC="${STARTUP_DELAY_SEC:-0}"
STRESS_MODE="${STRESS_MODE:-none}"

set +u
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
set -u

mkdir -p "$ROOT/data/results"

echo "Starting concurrent EdgeAI-ROS pipeline."
echo "policy=$POLICY vit_deadline_ms=$VIT_DEADLINE_MS llm_deadline_ms=$LLM_DEADLINE_MS"
echo "vit_csv_path=$VIT_CSV_PATH"
echo "llm_csv_path=$LLM_CSV_PATH"

ros2 run edgeai_ros edgeai_logger \
  --ros-args \
  -p latency_topic:=/repvit_latency \
  -p csv_path:="$VIT_CSV_PATH" &
VIT_LOGGER_PID=$!

ros2 run edgeai_ros edgeai_logger \
  --ros-args \
  -p latency_topic:=/llm_metrics \
  -p csv_path:="$LLM_CSV_PATH" &
LLM_LOGGER_PID=$!

ros2 run edgeai_ros adaptive_repvit_node \
  --ros-args \
  -p policy:="$POLICY" \
  -p deadline_ms:="$VIT_DEADLINE_MS" \
  -p checkpoint_dir:="$ROOT/checkpoints" \
  -p repvit_path:="${REPVIT_PATH:-/workspace/OS2026/RepViT}" &
VIT_PID=$!

ros2 run edgeai_ros adaptive_llm_node \
  --ros-args \
  -p policy:="$POLICY" \
  -p deadline_ms:="$LLM_DEADLINE_MS" \
  -p model:="$LLM_MODEL" \
  -p device:="$LLM_DEVICE" \
  -p local_files_only:="$LOCAL_FILES_ONLY" &
LLM_PID=$!

STRESS_PID=""
if [[ "$STRESS_MODE" != "none" ]]; then
  ros2 run edgeai_ros stress_node --ros-args -p mode:="$STRESS_MODE" &
  STRESS_PID=$!
fi

if awk "BEGIN {exit !($STARTUP_DELAY_SEC > 0)}"; then
  echo "Waiting ${STARTUP_DELAY_SEC}s before starting publishers..."
  sleep "$STARTUP_DELAY_SEC"
fi

ros2 run edgeai_ros image_publisher \
  --ros-args \
  -p image_dir:="$IMAGE_DIR" \
  -p period_sec:="$VIT_PERIOD_SEC" &
IMAGE_PID=$!

ros2 run edgeai_ros prompt_publisher \
  --ros-args \
  -p period_sec:="$PROMPT_PERIOD_SEC" \
  -p prompt_repeat:="$PROMPT_REPEAT" &
PROMPT_PID=$!

trap 'kill $PROMPT_PID $IMAGE_PID $LLM_PID $VIT_PID $LLM_LOGGER_PID $VIT_LOGGER_PID ${STRESS_PID:-} 2>/dev/null || true' EXIT
if awk "BEGIN {exit !($DURATION_SEC > 0)}"; then
  sleep "$DURATION_SEC"
  kill $PROMPT_PID $IMAGE_PID $LLM_PID $VIT_PID $LLM_LOGGER_PID $VIT_LOGGER_PID ${STRESS_PID:-} 2>/dev/null || true
  wait 2>/dev/null || true
else
  wait
fi
