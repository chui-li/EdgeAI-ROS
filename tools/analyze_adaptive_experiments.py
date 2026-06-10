from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def load_labels(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def label_for(pred_class, labels):
    try:
        idx = int(pred_class)
    except Exception:
        return ""
    if 0 <= idx < len(labels):
        return labels[idx]
    return ""


def infer_name(path: Path):
    stem = path.stem
    for token in [
        "predictive_adaptive_",
        "static_large_",
        "static_small_",
        "rule_adaptive_",
    ]:
        if token in stem:
            return stem.split(token, 1)[1]
    return stem


def summarize_file(path: Path, labels, warmup_frames: int):
    df = pd.read_csv(path)
    if warmup_frames > 0 and "frame" in df.columns:
        df = df[pd.to_numeric(df["frame"], errors="coerce") > warmup_frames].copy()

    for col in ["infer_ms", "e2e_ms", "deadline_miss", "level", "image_size", "fallback_model"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "pred_class" in df.columns:
        df["label"] = df["pred_class"].map(lambda value: label_for(value, labels))

    rows = len(df)
    deadline = df["deadline_ms"].iloc[0] if rows and "deadline_ms" in df.columns else ""
    policy = df["policy"].iloc[0] if rows and "policy" in df.columns else ""
    source = df["model_source"].mode().iloc[0] if rows and "model_source" in df.columns and not df["model_source"].mode().empty else ""
    device = df["device"].mode().iloc[0] if rows and "device" in df.columns and not df["device"].mode().empty else ""
    fallback_rows = int((df.get("fallback_model", pd.Series(dtype=float)) != 0).sum()) if rows else 0

    out = {
        "experiment": infer_name(path),
        "file": path.name,
        "policy": policy,
        "device": device,
        "deadline_ms": deadline,
        "rows_after_warmup": rows,
        "warmup_frames_excluded": warmup_frames,
        "fallback_rows": fallback_rows,
        "model_source": source,
        "deadline_misses": int(df["deadline_miss"].sum()) if rows and "deadline_miss" in df.columns else 0,
        "deadline_miss_rate": df["deadline_miss"].mean() if rows and "deadline_miss" in df.columns else 0.0,
        "avg_infer_ms": df["infer_ms"].mean() if rows and "infer_ms" in df.columns else 0.0,
        "median_infer_ms": df["infer_ms"].median() if rows and "infer_ms" in df.columns else 0.0,
        "p95_infer_ms": df["infer_ms"].quantile(0.95) if rows and "infer_ms" in df.columns else 0.0,
        "avg_e2e_ms": df["e2e_ms"].mean() if rows and "e2e_ms" in df.columns else 0.0,
        "p95_e2e_ms": df["e2e_ms"].quantile(0.95) if rows and "e2e_ms" in df.columns else 0.0,
        "avg_level": df["level"].mean() if rows and "level" in df.columns else 0.0,
        "avg_image_size": df["image_size"].mean() if rows and "image_size" in df.columns else 0.0,
    }

    if rows and "action" in df.columns:
        counts = df["action"].value_counts()
        for action in ["accept", "degrade", "defer", "drop", "reject"]:
            out[f"action_{action}"] = int(counts.get(action, 0))

    if rows and "level" in df.columns:
        counts = df["level"].value_counts()
        for level in [0, 1, 2]:
            out[f"level_{level}_rows"] = int(counts.get(level, 0))

    return out, df


def main():
    parser = argparse.ArgumentParser(description="Summarize adaptive RepViT experiment CSV files.")
    parser.add_argument("csv", nargs="+", help="CSV files to summarize")
    parser.add_argument("--labels", default="data/imagenet_classes.txt")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--out", default="data/results/adaptive_experiment_summary.csv")
    parser.add_argument("--pred-out", default="data/results/adaptive_prediction_summary.csv")
    args = parser.parse_args()

    labels = load_labels(Path(args.labels))
    summaries = []
    pred_rows = []

    for item in args.csv:
        path = Path(item)
        summary, df = summarize_file(path, labels, args.warmup_frames)
        summaries.append(summary)
        if not df.empty and {"image", "pred_class", "label"}.issubset(df.columns):
            grouped = df.groupby(["image", "pred_class", "label"], dropna=False).size().reset_index(name="rows")
            grouped.insert(0, "experiment", summary["experiment"])
            grouped.insert(1, "file", path.name)
            pred_rows.append(grouped)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.out, index=False)
    print(summary_df.round(3).to_string(index=False))
    print(f"Saved summary to {args.out}")

    if pred_rows:
        pred_df = pd.concat(pred_rows, ignore_index=True)
        pred_df.to_csv(args.pred_out, index=False)
        print(f"Saved prediction summary to {args.pred_out}")


if __name__ == "__main__":
    main()
