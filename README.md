# EdgeAI-ROS

**EdgeAI-ROS: An OS-Aware Adaptive ROS 2 Runtime for RepViT Inference under Edge Resource Constraints**

EdgeAI-ROS combines a ROS 2 RepViT image inference pipeline with OS-aware runtime control. It is designed for experiments where AI inference latency is affected by CPU scheduling, memory pressure, page faults, context switches, I/O pressure, queue pressure, concurrent LLM load, and real-time deadlines.

The runtime subscribes to image frames and LLM prompts, monitors OS state, predicts latency with EWMA, selects a runtime configuration, runs inference or generation, and writes CSV logs for later analysis.

```text
image_publisher
    |
    | /image_raw
    v
adaptive_repvit_node
    | OSMonitor + EWMALatencyPredictor + PredictiveScheduler + AdmissionController
    |
    | /repvit_latency
    | /edgeai_os_state
    | /edgeai_scheduler_decision
    v
edgeai_logger
    |
    v
data/results/*.csv
```

LLM support uses the same runtime ideas:

```text
prompt_publisher
    |
    | /llm_prompt
    v
adaptive_llm_node
    |
    | /llm_metrics
    | /llm_scheduler_decision
    v
edgeai_logger
    |
    v
data/results/*llm*.csv
```

## Repository Layout

```text
EdgeAI-ROS/
+-- ros2_ws/src/edgeai_ros/
|   +-- edgeai_ros/
|   |   +-- adaptive_repvit_node.py
|   |   +-- adaptive_llm_node.py
|   |   +-- image_publisher.py
|   |   +-- prompt_publisher.py
|   |   +-- edgeai_logger.py
|   |   +-- stress_node.py
|   |   +-- repvit_backend.py
|   |   +-- llm_worker.py
|   |   +-- monitor.py
|   |   +-- predictor.py
|   |   +-- scheduler.py
|   |   +-- admission.py
|   |   +-- runtime.py
|   +-- launch/
|   +-- package.xml
|   +-- setup.py
+-- configs/
+-- scripts/
+-- tools/
+-- data/test_images/
+-- data/results/
+-- checkpoints/
+-- reports/
+-- docker/
+-- download_model.py
+-- requirements.txt
```

## Runtime Policies

Supported policies:

- `static_large`: always choose the highest-quality config.
- `static_small`: always choose the lightest config.
- `rule_adaptive`: downgrade or upgrade using pressure and deadline feedback.
- `predictive_adaptive`: use EWMA latency prediction, OS pressure, deadline feasibility, and utility scoring.

Default RepViT adaptive configs:

```text
level 0: repvit_m0_9, image_size=160, quality_score=0.72
level 1: repvit_m0_9, image_size=192, quality_score=0.86
level 2: repvit_m0_9, image_size=224, quality_score=1.00
```

By default this adapts image resolution. To switch between different RepViT model variants, pass multiple model names through the `MODELS` environment variable or the ROS `models` parameter.

Default LLM configs:

```text
level 0: context_length=512,  max_new_tokens=32,  quality_score=0.68
level 1: context_length=1024, max_new_tokens=64,  quality_score=0.84
level 2: context_length=2048, max_new_tokens=128, quality_score=1.00
```

For the LLM node, `deadline_ms` is treated as a TPOT target in milliseconds per generated token.

## Model Loading

`adaptive_repvit_node` tries model sources in this order:

1. Official RepViT code through `REPViT_PATH`, `REPVIT_PATH`, `EDGEAI_REPVIT_PATH`, or known paths such as `/workspace/OS2026/RepViT`.
2. `timm.create_model(...)`.
3. `TinyFallbackCNN`.

The fallback CNN keeps OS and ROS experiments runnable without checkpoints, but its predictions are not meaningful for accuracy claims.

Checkpoints can be placed in:

```text
checkpoints/
```

Example checkpoint names:

```text
checkpoints/repvit_m0_9.pth
checkpoints/repvit_m0_9_distill_300e.pth
```

For official RepViT experiments, set:

```bash
export REPVIT_PATH=/workspace/OS2026/RepViT
```

`adaptive_llm_node` tries to load a Hugging Face causal language model. The default is:

```text
sshleifer/tiny-gpt2
```

If `transformers` is unavailable, the model is not cached, or network access is disabled, the node falls back to an offline lightweight generator. The fallback keeps OS scheduling experiments runnable, but language quality should not be evaluated.

To download the Gemma model used in later local LLM experiments:

```bash
python3 download_model.py
```

This downloads `google/gemma-3-1b-it` into `./gemma-3-1b-it`.

## Environment Setup

Use Docker or WSL2 Ubuntu 22.04 with ROS 2 Humble. On Windows, the recommended path is Docker or WSL2 because `rclpy` is provided by ROS 2, not by `pip`.

Build and enter the Docker image:

```bash
./scripts/docker_build.sh
./scripts/docker_run.sh
```

Inside the container, install Python dependencies if needed, create sample images, build the workspace, and source it:

```bash
cd /workspace/EdgeAI-ROS
python3 -m pip install -r requirements.txt
python3 tools/make_test_image.py
./scripts/build_ws.sh
source ros2_ws/install/setup.bash
mkdir -p data/results
```

If you are not using Docker, run the same commands from a ROS 2 Humble terminal after installing the ROS dependencies declared in `ros2_ws/src/edgeai_ros/package.xml`.

## Quick Smoke Test

Run one short RepViT pipeline first. The pipeline scripts run until stopped, so `timeout` is used for repeatable experiments. Exit code `124` from `timeout` is expected.

```bash
env \
  DEVICE=auto \
  CSV_PATH=data/results/smoke_predictive_200ms.csv \
  timeout 60s ./scripts/run_pipeline.sh predictive_adaptive 200.0 || true
```

Check the CSV:

```bash
python3 tools/summarize_latency.py data/results/smoke_predictive_200ms.csv
```

Run one short LLM-only smoke test:

```bash
env \
  DEVICE=auto \
  LLM_MODEL=sshleifer/tiny-gpt2 \
  CSV_PATH=data/results/llm_smoke_80ms.csv \
  DURATION_SEC=60 \
  ./scripts/run_llm_pipeline.sh predictive_adaptive 80.0
```

## Full RepViT CPU/GPU Experiments

This section reproduces the main CPU/GPU RepViT runtime matrix used by `reports/final_cpu_gpu_repvit_research_report_zh.md`.

Before running GPU experiments, confirm CUDA is visible:

```bash
nvidia-smi
```

Use these shared settings unless you intentionally want a different run length:

```bash
export RUN_SEC=90
export IMAGE_DIR=/workspace/EdgeAI-ROS/data/test_images
export REPVIT_PATH=/workspace/OS2026/RepViT
```

### 1. Predictive Adaptive Deadline Sweep

Runs `predictive_adaptive` at 250 ms, 200 ms, and 150 ms on CPU and GPU.

```bash
for device in cpu cuda; do
  if [ "$device" = "cuda" ]; then label=gpu; else label=cpu; fi
  for deadline in 250 200 150; do
    env \
      DEVICE="$device" \
      IMAGE_DIR="$IMAGE_DIR" \
      CSV_PATH="data/results/final_${label}_predictive_${deadline}ms.csv" \
      timeout "${RUN_SEC}s" ./scripts/run_pipeline.sh predictive_adaptive "${deadline}.0" \
      > "data/results/final_${label}_predictive_${deadline}ms.log" 2>&1 || true
  done
done
```

### 2. Policy Comparison At 200 ms

Runs the static and adaptive policies at the same deadline.

```bash
for device in cpu cuda; do
  if [ "$device" = "cuda" ]; then label=gpu; else label=cpu; fi
  for policy in static_large static_small rule_adaptive predictive_adaptive; do
    env \
      DEVICE="$device" \
      IMAGE_DIR="$IMAGE_DIR" \
      CSV_PATH="data/results/final_${label}_${policy}_200ms.csv" \
      timeout "${RUN_SEC}s" ./scripts/run_pipeline.sh "$policy" 200.0 \
      > "data/results/final_${label}_${policy}_200ms.log" 2>&1 || true
  done
done
```

### 3. Stress Experiments At 200 ms

Runs `predictive_adaptive` with CPU stress and mixed CPU/memory/I/O stress.

```bash
for device in cpu cuda; do
  if [ "$device" = "cuda" ]; then label=gpu; else label=cpu; fi
  for stress in cpu mixed; do
    env \
      DEVICE="$device" \
      IMAGE_DIR="$IMAGE_DIR" \
      CSV_PATH="data/results/final_${label}_stress_${stress}_200ms.csv" \
      timeout "${RUN_SEC}s" ./scripts/run_with_stress.sh "$stress" predictive_adaptive vit 200.0 \
      > "data/results/final_${label}_stress_${stress}_200ms.log" 2>&1 || true
  done
done
```

### 4. Summarize The Main RepViT Matrix

```bash
python3 tools/analyze_adaptive_experiments.py \
  data/results/final_*_*.csv \
  --out data/results/final_cpu_gpu_summary.csv \
  --pred-out data/results/final_cpu_gpu_prediction_summary.csv
```

The key output files are:

```text
data/results/final_cpu_gpu_summary.csv
data/results/final_cpu_gpu_prediction_summary.csv
```

## RepViT Model Family Benchmark

This benchmark compares RepViT-M0.6, M0.9, M1.0, M1.1, M1.5, and M2.3 at 200 ms. It keeps all three runtime levels on the same model and resolution so the result is a model-family comparison rather than a resolution-adaptation run.

```bash
export RUN_SEC=90
export IMAGE_DIR=/workspace/EdgeAI-ROS/data/test_images

for device in cpu cuda; do
  if [ "$device" = "cuda" ]; then label=gpu; else label=cpu; fi
  for model in repvit_m0_6 repvit_m0_9 repvit_m1_0 repvit_m1_1 repvit_m1_5 repvit_m2_3; do
    env \
      DEVICE="$device" \
      MODELS="$model,$model,$model" \
      IMAGE_SIZES="224,224,224" \
      QUALITY_SCORES="1.0,1.0,1.0" \
      IMAGE_DIR="$IMAGE_DIR" \
      CSV_PATH="data/results/model_benchmark_${label}_${model}_200ms.csv" \
      timeout "${RUN_SEC}s" ./scripts/run_pipeline.sh static_large 200.0 \
      > "data/results/model_benchmark_${label}_${model}_200ms.log" 2>&1 || true
  done
done
```

Analyze the model family:

```bash
python3 tools/analyze_repvit_model_family.py \
  --pattern "data/results/model_benchmark_*_200ms.csv" \
  --summary-out data/results/repvit_model_family_summary.csv \
  --speedup-out data/results/repvit_model_family_speedup.csv \
  --pred-out data/results/repvit_model_family_predictions.csv \
  --report-out reports/repvit_model_family_benchmark_zh.md
```

## ImageNet Subset Accuracy And Latency

For accuracy-oriented runs, put real labeled images under `data/imagenet_test_images` or use a prepared subset under `data/imagenet_val_subset_1_per_class/images`.

Resize real-image inputs:

```bash
python3 tools/prepare_imagenet_images.py \
  --src data/imagenet_test_images \
  --dst data/imagenet_test_images_resized \
  --max-side 512
```

Run a RepViT-M0.9 GPU static-large ImageNet-style pass:

```bash
env \
  DEVICE=cuda \
  MODELS="repvit_m0_9,repvit_m0_9,repvit_m0_9" \
  IMAGE_SIZES="224,224,224" \
  QUALITY_SCORES="1.0,1.0,1.0" \
  IMAGE_DIR=/workspace/EdgeAI-ROS/data/imagenet_test_images_resized \
  CSV_PATH=data/results/edgeai_ros_imagenet_resized_predictive_adaptive_200ms.csv \
  timeout 120s ./scripts/run_pipeline.sh predictive_adaptive 200.0 || true
```

If you have the 1-per-class validation subset available, run the model-family subset benchmark:

```bash
for model in repvit_m0_6 repvit_m0_9 repvit_m1_0 repvit_m1_1 repvit_m1_5 repvit_m2_3; do
  env \
    DEVICE=cuda \
    MODELS="$model,$model,$model" \
    IMAGE_SIZES="224,224,224" \
    QUALITY_SCORES="1.0,1.0,1.0" \
    IMAGE_DIR=/workspace/EdgeAI-ROS/data/imagenet_val_subset_1_per_class/images \
    CSV_PATH="data/results/imagenet_val_subset_gpu_${model}_static_large_200ms.csv" \
    timeout 900s ./scripts/run_pipeline.sh static_large 200.0 \
    > "data/results/imagenet_val_subset_gpu_${model}_static_large_200ms.log" 2>&1 || true
done
```

Analyze ImageNet subset outputs:

```bash
python3 tools/analyze_imagenet_val_subset.py \
  --csv data/results/imagenet_val_subset_gpu_repvit_m0_9_static_large_200ms.csv \
  --summary-out data/results/imagenet_val_subset_gpu_repvit_m0_9_summary.csv \
  --pred-out data/results/imagenet_val_subset_gpu_repvit_m0_9_predictions.csv

python3 tools/analyze_imagenet_val_model_family.py \
  --pattern "data/results/imagenet_val_subset_gpu_repvit_*_static_large_200ms.csv" \
  --summary-out data/results/imagenet_val_subset_gpu_repvit_family_summary.csv \
  --pred-out data/results/imagenet_val_subset_gpu_repvit_family_predictions.csv \
  --report-out reports/imagenet_val_subset_repvit_family_zh.md
```

## Runtime Decomposition And Recommended Runtime

Use the recommended runtime script to test model/device switching with deadline-aware queueing:

```bash
env \
  CSV_PATH=data/results/modified_runtime_recommended_200ms.csv \
  timeout 180s ./scripts/run_recommended_runtime.sh predictive_adaptive 200.0 || true
```

Summarize runtime decomposition:

```bash
python3 tools/summarize_runtime_decomposition.py \
  data/results/*runtime*.csv \
  --out data/results/modified_runtime_recommended_200ms_summary.csv
```

Important recommended-runtime settings:

```text
MODELS=repvit_m0_6,repvit_m0_9,repvit_m1_1,repvit_m1_5,repvit_m2_3
IMAGE_SIZES=160,192,224,224,224
DEVICES=cpu,cpu,cuda,cuda,cuda
QOS_DEPTH=1
QOS_RELIABILITY=best_effort
QUEUE_POLICY=deadline_drop
PERIOD_SEC=0.2
```

## LLM-Only Experiments

Use local models when possible. For Llama or Gemma, set `LLM_MODEL` to the local directory.

Gemma example:

```bash
export LLM_MODEL=/workspace/EdgeAI-ROS/gemma-3-1b-it
export LOCAL_FILES_ONLY=true
```

Run LLM-only experiments across stress modes and deadlines:

```bash
for stress in none cpu; do
  for deadline in 200 150 100 80; do
    env \
      DEVICE=cuda \
      LLM_MODEL="$LLM_MODEL" \
      LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY" \
      STRESS_MODE="$stress" \
      CSV_PATH="data/results/gemma3_1b_only_${stress}_${deadline}p0ms.csv" \
      DURATION_SEC=60 \
      ./scripts/run_llm_pipeline.sh predictive_adaptive "${deadline}.0"
  done
done
```

Analyze LLM-only outputs:

```bash
python3 tools/analyze_llm_experiments.py \
  data/results/gemma3_1b_only_*.csv \
  --out data/results/gemma3_1b_only_summary.csv
```

## Concurrent ViT And LLM Experiments

Run RepViT and LLM together to measure interference. The ViT deadline is usually fixed at 200 ms while the LLM deadline is swept.

```bash
export LLM_MODEL=/workspace/EdgeAI-ROS/gemma-3-1b-it
export LOCAL_FILES_ONLY=true
export DURATION_SEC=90

for stress in none mixed; do
  for llm_deadline in 200 150 100; do
    env \
      DEVICE=cuda \
      LLM_DEVICE=cuda \
      LLM_MODEL="$LLM_MODEL" \
      LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY" \
      STRESS_MODE="$stress" \
      DURATION_SEC="$DURATION_SEC" \
      VIT_CSV_PATH="data/results/gemma3_concurrent_${stress}_llm${llm_deadline}_vit.csv" \
      LLM_CSV_PATH="data/results/gemma3_concurrent_${stress}_llm${llm_deadline}_llm.csv" \
      ./scripts/run_concurrent_pipeline.sh predictive_adaptive 200.0 "${llm_deadline}.0"
  done
done
```

Analyze concurrent outputs:

```bash
python3 tools/analyze_adaptive_experiments.py \
  data/results/gemma3_concurrent_*_vit.csv \
  --out data/results/gemma3_concurrent_vit_summary.csv \
  --pred-out data/results/gemma3_concurrent_vit_prediction_summary.csv

python3 tools/analyze_llm_experiments.py \
  data/results/gemma3_concurrent_*_llm.csv \
  --out data/results/gemma3_concurrent_llm_summary.csv
```

## Manual ROS 2 Commands

Use manual commands when debugging one node at a time.

Terminal 1:

```bash
ros2 run edgeai_ros edgeai_logger \
  --ros-args -p csv_path:=/workspace/EdgeAI-ROS/data/results/manual_repvit.csv
```

Terminal 2:

```bash
ros2 run edgeai_ros adaptive_repvit_node \
  --ros-args \
  -p policy:=predictive_adaptive \
  -p deadline_ms:=200.0 \
  -p device:=cuda \
  -p checkpoint_dir:=/workspace/EdgeAI-ROS/checkpoints \
  -p repvit_path:=/workspace/OS2026/RepViT
```

Terminal 3:

```bash
ros2 run edgeai_ros image_publisher \
  --ros-args \
  -p image_dir:=/workspace/EdgeAI-ROS/data/test_images \
  -p period_sec:=0.5
```

Optional stress node:

```bash
ros2 run edgeai_ros stress_node --ros-args -p mode:=mixed
```

Manual LLM commands:

```bash
ros2 run edgeai_ros edgeai_logger \
  --ros-args \
  -p latency_topic:=/llm_metrics \
  -p csv_path:=/workspace/EdgeAI-ROS/data/results/manual_llm.csv
```

```bash
ros2 run edgeai_ros adaptive_llm_node \
  --ros-args \
  -p policy:=predictive_adaptive \
  -p deadline_ms:=80.0 \
  -p model:=sshleifer/tiny-gpt2 \
  -p device:=auto
```

```bash
ros2 run edgeai_ros prompt_publisher \
  --ros-args -p period_sec:=5.0
```

## Output Metrics

The logger writes CSV rows with fields such as:

- `infer_ms`
- `e2e_ms`
- `predicted_latency_ms`
- `deadline_ms`
- `deadline_miss`
- `deadline_miss_rate`
- `pressure_score`
- `cpu_percent`
- `memory_percent`
- `process_rss_mb`
- `page_faults_delta`
- `ctx_switches_delta`
- `model`
- `level`
- `image_size`
- `model_switched`
- `model_switch_count`
- `context_length`
- `max_new_tokens`
- `ttft_ms`
- `tpot_ms`
- `tokens_per_sec`
- `output_tokens`
- `fallback_model`
- `model_source`

Useful checks:

```bash
python3 tools/summarize_latency.py data/results/final_gpu_predictive_200ms.csv
python3 tools/plot_latency.py data/results/final_gpu_predictive_200ms.csv
```

## Expected Reports And Result Files

After a full run, the main artifacts should include:

```text
data/results/final_cpu_gpu_summary.csv
data/results/final_cpu_gpu_prediction_summary.csv
data/results/repvit_model_family_summary.csv
data/results/repvit_model_family_speedup.csv
data/results/repvit_model_family_predictions.csv
data/results/imagenet_val_subset_gpu_repvit_family_summary.csv
data/results/imagenet_val_subset_gpu_repvit_family_predictions.csv
data/results/gemma3_1b_only_summary.csv
data/results/gemma3_concurrent_vit_summary.csv
data/results/gemma3_concurrent_llm_summary.csv
reports/final_cpu_gpu_repvit_research_report_zh.md
reports/repvit_model_family_benchmark_zh.md
reports/edgeai_ros_repvit_complete_research_report_zh.md
```

## Troubleshooting

If `rclpy` cannot be imported, use a ROS 2 Humble environment. Installing `rclpy` with `pip` is not enough.

If RepViT falls back to `TinyFallbackCNN`, check `REPVIT_PATH`, `checkpoint_dir`, and checkpoint file names.

If CUDA is not used, check `nvidia-smi`, Docker GPU flags, `DEVICE=cuda`, and whether PyTorch sees CUDA.

If CSV files are empty, increase `RUN_SEC`, check that images exist in `IMAGE_DIR`, and make sure `edgeai_logger` is subscribed to the correct topic.

If `timeout` returns exit code `124`, that is normal for these foreground ROS pipeline scripts.

## Relationship To Existing Projects

- `OS2026` contributes the ROS 2 image pipeline and RepViT model source.
- `os-project` contributes OS monitoring, latency prediction, adaptive scheduling, admission control, and stress injection.
- `EdgeAI-ROS` turns these into one ROS 2 runtime experiment framework.
