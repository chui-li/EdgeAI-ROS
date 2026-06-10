from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NUMERIC_COLS = [
    "ttft_ms",
    "tpot_ms",
    "tokens_per_sec",
    "output_tokens",
    "e2e_ms",
    "predicted_latency_ms",
    "deadline_ms",
    "deadline_miss",
    "pressure_score",
    "queue_size",
    "deferred_requests",
    "cpu_percent",
    "memory_percent",
    "process_rss_mb",
    "page_faults_delta",
    "ctx_switches_delta",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").dropna()


def summarize_file(path: Path) -> dict:
    df = pd.read_csv(path)
    row: dict[str, object] = {"file": path.name, "rows": len(df)}

    for col in NUMERIC_COLS:
        s = _num(df, col)
        if s.empty:
            continue
        row[f"avg_{col}"] = s.mean()
        row[f"p50_{col}"] = s.quantile(0.50)
        row[f"p95_{col}"] = s.quantile(0.95)
        row[f"p99_{col}"] = s.quantile(0.99)
        row[f"max_{col}"] = s.max()

    miss = _num(df, "deadline_miss")
    if not miss.empty:
        row["miss_rate"] = miss.mean()
        row["misses"] = int(miss.sum())

    for col in ["level", "context_length", "max_new_tokens", "fallback_model"]:
        s = _num(df, col)
        if not s.empty:
            row[f"avg_{col}"] = s.mean()

    if "level" in df.columns:
        levels = _num(df, "level")
        if not levels.empty:
            counts = levels.astype(int).value_counts().sort_index()
            row["level_distribution"] = "; ".join(f"L{k}:{v}" for k, v in counts.items())

    if "action" in df.columns:
        actions = df["action"].dropna().astype(str).value_counts()
        if not actions.empty:
            row["action_distribution"] = "; ".join(f"{k}:{v}" for k, v in actions.items())

    for col in ["device", "model", "policy"]:
        if col in df.columns and len(df[col].dropna()):
            row[col] = str(df[col].dropna().mode().iloc[0])

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EdgeAI-ROS LLM CSV files.")
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--out", default="data/results/llm_experiment_summary.csv")
    args = parser.parse_args()

    rows = [summarize_file(Path(p)) for p in args.csv]
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    with pd.option_context("display.max_columns", 80, "display.width", 220):
        print(table.round(3).to_string(index=False))
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
