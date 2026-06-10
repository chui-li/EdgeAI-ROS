from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NUMERIC_COLS = [
    "image_load_ms",
    "publish_overhead_ms",
    "logger_receive_ms",
    "logging_ms",
    "preprocess_ms",
    "infer_ms",
    "postprocess_ms",
    "model_get_ms",
    "model_load_latency_ms",
    "capture_age_at_receive_ms",
    "e2e_ms",
    "freshness_ms",
    "non_model_overhead_ms",
    "inference_ratio",
    "preprocess_ratio",
    "communication_ratio",
    "predicted_latency_ms",
    "deadline_ms",
    "deadline_miss",
    "pressure_score",
    "queue_length",
    "queue_size",
    "dropped_frames",
    "deferred_frames",
    "accepted_frames",
    "arrival_rate_hz",
    "service_rate_hz",
    "utilization_rho",
    "effective_fps",
    "model_cache_miss",
    "model_switched",
    "model_switch_count",
    "deadline_miss_after_switch",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "cpu_percent",
    "memory_percent",
    "page_faults_delta",
    "ctx_switches_delta",
    "cpu_migration_delta",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize deadline decomposition CSVs.")
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--out", default="data/results/runtime_decomposition_summary.csv")
    args = parser.parse_args()

    rows = []
    for csv_path in args.csv:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        data = df[pd.to_numeric(df["frame"], errors="coerce") > args.warmup_frames].copy()
        if data.empty:
            continue

        group_cols = [c for c in ["policy", "model", "config_device", "device", "queue_policy"] if c in data.columns]
        grouped = data.groupby(group_cols, dropna=False) if group_cols else [((), data)]

        for key, g in grouped:
            item = {"file": Path(csv_path).name, "rows": len(g)}
            if group_cols:
                if not isinstance(key, tuple):
                    key = (key,)
                item.update(dict(zip(group_cols, key)))
            item.update(
                {
                    "deadline_misses": int(g["deadline_miss"].sum()) if "deadline_miss" in g else 0,
                    "deadline_miss_rate": float(g["deadline_miss"].mean()) if "deadline_miss" in g else 0.0,
                    "accepts": int(g.get("action", pd.Series(dtype=str)).isin(["accept", "degrade"]).sum())
                    if "action" in g
                    else 0,
                    "drops": int((g.get("action", pd.Series(dtype=str)) == "drop").sum()) if "action" in g else 0,
                    "defers": int((g.get("action", pd.Series(dtype=str)) == "defer").sum()) if "action" in g else 0,
                    "avg_image_load_ms": g.get("image_load_ms", pd.Series(dtype=float)).mean(),
                    "avg_publish_overhead_ms": g.get("publish_overhead_ms", pd.Series(dtype=float)).mean(),
                    "avg_preprocess_ms": g.get("preprocess_ms", pd.Series(dtype=float)).mean(),
                    "avg_infer_ms": g.get("infer_ms", pd.Series(dtype=float)).mean(),
                    "p95_infer_ms": g.get("infer_ms", pd.Series(dtype=float)).quantile(0.95),
                    "avg_postprocess_ms": g.get("postprocess_ms", pd.Series(dtype=float)).mean(),
                    "avg_logger_receive_ms": g.get("logger_receive_ms", pd.Series(dtype=float)).mean(),
                    "avg_logging_ms": g.get("logging_ms", pd.Series(dtype=float)).mean(),
                    "avg_model_get_ms": g.get("model_get_ms", pd.Series(dtype=float)).mean(),
                    "cold_loads": int((g.get("model_cache_hit", pd.Series(dtype=float)) == 0).sum())
                    if "model_cache_hit" in g
                    else 0,
                    "avg_model_load_latency_ms": g.get("model_load_latency_ms", pd.Series(dtype=float)).mean(),
                    "cache_misses": int(g.get("model_cache_miss", pd.Series(dtype=float)).sum())
                    if "model_cache_miss" in g
                    else 0,
                    "model_switches": int(g.get("model_switched", pd.Series(dtype=float)).sum())
                    if "model_switched" in g
                    else 0,
                    "deadline_miss_after_switch": int(g.get("deadline_miss_after_switch", pd.Series(dtype=float)).max())
                    if "deadline_miss_after_switch" in g
                    else 0,
                    "avg_non_model_overhead_ms": g.get("non_model_overhead_ms", pd.Series(dtype=float)).mean(),
                    "p95_non_model_overhead_ms": g.get("non_model_overhead_ms", pd.Series(dtype=float)).quantile(0.95),
                    "avg_inference_ratio": g.get("inference_ratio", pd.Series(dtype=float)).mean(),
                    "avg_preprocess_ratio": g.get("preprocess_ratio", pd.Series(dtype=float)).mean(),
                    "avg_communication_ratio": g.get("communication_ratio", pd.Series(dtype=float)).mean(),
                    "avg_e2e_ms": g.get("e2e_ms", pd.Series(dtype=float)).mean(),
                    "p95_e2e_ms": g.get("e2e_ms", pd.Series(dtype=float)).quantile(0.95),
                    "avg_freshness_ms": g.get("freshness_ms", pd.Series(dtype=float)).mean(),
                    "p95_freshness_ms": g.get("freshness_ms", pd.Series(dtype=float)).quantile(0.95),
                    "avg_arrival_rate_hz": g.get("arrival_rate_hz", pd.Series(dtype=float)).mean(),
                    "avg_service_rate_hz": g.get("service_rate_hz", pd.Series(dtype=float)).mean(),
                    "avg_utilization_rho": g.get("utilization_rho", pd.Series(dtype=float)).mean(),
                    "avg_effective_fps": g.get("effective_fps", pd.Series(dtype=float)).mean(),
                    "avg_ctx_switches_delta": g.get("ctx_switches_delta", pd.Series(dtype=float)).mean(),
                    "cpu_migrations": int(g.get("cpu_migration_delta", pd.Series(dtype=float)).sum())
                    if "cpu_migration_delta" in g
                    else 0,
                    "avg_page_faults_delta": g.get("page_faults_delta", pd.Series(dtype=float)).mean(),
                    "avg_gpu_memory_allocated_mb": g.get("gpu_memory_allocated_mb", pd.Series(dtype=float)).mean(),
                    "avg_gpu_memory_reserved_mb": g.get("gpu_memory_reserved_mb", pd.Series(dtype=float)).mean(),
                }
            )
            rows.append(item)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
