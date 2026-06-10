#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OS2026_ROOT="${OS2026_ROOT:-/mnt/usr1/azure005/OS2026/OS2026}"

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

docker run --rm -it \
  "${GPU_ARGS[@]}" \
  --network host \
  -v "$ROOT":/workspace/EdgeAI-ROS \
  -v "$OS2026_ROOT":/workspace/OS2026:ro \
  -w /workspace/EdgeAI-ROS \
  edgeai-ros:humble \
  bash

