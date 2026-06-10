#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${1:-predictive_adaptive}"
DEADLINE_MS="${2:-80.0}"
CSV_PATH="${CSV_PATH:-$ROOT/data/results/edgeai_ros_llm_${POLICY}_${DEADLINE_MS}ms.csv}"
MODEL="${LLM_MODEL:-sshleifer/tiny-gpt2}"
DEVICE="${DEVICE:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"
PROMPT_PERIOD_SEC="${PROMPT_PERIOD_SEC:-1.0}"
PROMPT_REPEAT="${PROMPT_REPEAT:-1}"
PROMPT_TEXT="${PROMPT_TEXT:-Explain why operating system resource management matters for edge AI inference with vision models and language models.}"
MAX_QUEUE="${MAX_QUEUE:-1}"
QOS_DEPTH="${QOS_DEPTH:-10}"
QOS_RELIABILITY="${QOS_RELIABILITY:-reliable}"
STARTUP_DELAY_SEC="${STARTUP_DELAY_SEC:-8}"
DURATION_SEC="${DURATION_SEC:-60}"
STRESS_MODE="${STRESS_MODE:-none}"

set +u
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
set -u

mkdir -p "$ROOT/data/results"

echo "Starting LLM pipeline."
echo "policy=$POLICY deadline_ms=$DEADLINE_MS device=$DEVICE duration_sec=$DURATION_SEC stress=$STRESS_MODE"
echo "csv_path=$CSV_PATH"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

ros2 run edgeai_ros edgeai_logger \
  --ros-args \
  -p latency_topic:=/llm_metrics \
  -p csv_path:="$CSV_PATH" \
  -p qos_depth:="$QOS_DEPTH" \
  -p qos_reliability:="$QOS_RELIABILITY" &
PIDS+=("$!")

ros2 run edgeai_ros adaptive_llm_node \
  --ros-args \
  -p policy:="$POLICY" \
  -p deadline_ms:="$DEADLINE_MS" \
  -p model:="$MODEL" \
  -p device:="$DEVICE" \
  -p local_files_only:="$LOCAL_FILES_ONLY" \
  -p max_queue:="$MAX_QUEUE" &
PIDS+=("$!")

if [[ "$STRESS_MODE" != "none" ]]; then
  ros2 run edgeai_ros stress_node --ros-args -p mode:="$STRESS_MODE" &
  PIDS+=("$!")
fi

if awk "BEGIN {exit !($STARTUP_DELAY_SEC > 0)}"; then
  echo "Waiting ${STARTUP_DELAY_SEC}s before starting prompt publisher..."
  sleep "$STARTUP_DELAY_SEC"
fi

ros2 run edgeai_ros prompt_publisher \
  --ros-args \
  -p period_sec:="$PROMPT_PERIOD_SEC" \
  -p prompt_repeat:="$PROMPT_REPEAT" \
  -p prompt:="$PROMPT_TEXT" &
PIDS+=("$!")

sleep "$DURATION_SEC"
cleanup
wait 2>/dev/null || true
