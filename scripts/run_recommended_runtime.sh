#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${1:-predictive_adaptive}"
DEADLINE_MS="${2:-200.0}"
CSV_PATH="${CSV_PATH:-$ROOT/data/results/recommended_runtime_${POLICY}_${DEADLINE_MS}ms.csv}"

export MODELS="${MODELS:-repvit_m0_6,repvit_m0_9,repvit_m1_1,repvit_m1_5,repvit_m2_3}"
export IMAGE_SIZES="${IMAGE_SIZES:-160,192,224,224,224}"
export QUALITY_SCORES="${QUALITY_SCORES:-0.65,0.78,0.84,0.92,1.0}"
export DEVICES="${DEVICES:-cpu,cpu,cuda,cuda,cuda}"
export DEVICE="${DEVICE:-cuda}"
export CACHE_SIZE="${CACHE_SIZE:-5}"
export QOS_DEPTH="${QOS_DEPTH:-1}"
export QOS_RELIABILITY="${QOS_RELIABILITY:-best_effort}"
export QUEUE_POLICY="${QUEUE_POLICY:-deadline_drop}"
export PERIOD_SEC="${PERIOD_SEC:-0.2}"
export CSV_PATH

"$ROOT/scripts/run_pipeline.sh" "$POLICY" "$DEADLINE_MS"
