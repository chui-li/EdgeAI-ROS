#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t edgeai-ros:humble -f "$ROOT/docker/Dockerfile" "$ROOT"

