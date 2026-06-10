from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NUMERIC_COLS = [
    "infer_ms",
    "e2e_ms",
    "ttft_ms",
    "tpot_ms",
    "tokens_per_sec",
    "output_tokens",
    "predicted_latency_ms",
    "deadline_miss",
    "pressure_score",
    "cpu_percent",
    "memory_percent",
    "process_rss_mb",
    "page_faults_delta",
    "ctx_switches_delta",
]


def summarize(path: Path):
    df = pd.read_csv(path)
    row = {"file": path.name, "rows": len(df)}
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s):
            row[f"avg_{col}"] = s.mean()
            row[f"p99_{col}"] = s.quantile(0.99)
    if "level" in df.columns:
        s = pd.to_numeric(df["level"], errors="coerce").dropna()
        if len(s):
            row["avg_level"] = s.mean()
    if "image_size" in df.columns:
        s = pd.to_numeric(df["image_size"], errors="coerce").dropna()
        if len(s):
            row["avg_image_size"] = s.mean()
    if "context_length" in df.columns:
        s = pd.to_numeric(df["context_length"], errors="coerce").dropna()
        if len(s):
            row["avg_context_length"] = s.mean()
    if "max_new_tokens" in df.columns:
        s = pd.to_numeric(df["max_new_tokens"], errors="coerce").dropna()
        if len(s):
            row["avg_max_new_tokens"] = s.mean()
    if "model" in df.columns and len(df["model"].dropna()):
        row["most_selected_model"] = str(df["model"].dropna().mode().iloc[0])
    return row


def main():
    parser = argparse.ArgumentParser(description="Summarize EdgeAI-ROS latency CSV files.")
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--out", default="data/results/summary.csv")
    args = parser.parse_args()

    rows = [summarize(Path(path)) for path in args.csv]
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table)
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
