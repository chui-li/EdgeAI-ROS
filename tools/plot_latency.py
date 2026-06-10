from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_one(path: Path, out_dir: Path):
    df = pd.read_csv(path)
    x_col = "frame" if "frame" in df.columns else "request" if "request" in df.columns else None
    x = pd.to_numeric(df.get(x_col, pd.Series(range(len(df)))), errors="coerce").fillna(0)
    stem = path.stem
    for col in [
        "infer_ms",
        "e2e_ms",
        "ttft_ms",
        "tpot_ms",
        "tokens_per_sec",
        "predicted_latency_ms",
        "pressure_score",
        "deadline_miss",
        "level",
        "context_length",
        "max_new_tokens",
    ]:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().sum() == 0:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(x, y)
        plt.xlabel(x_col or "index")
        plt.ylabel(col)
        plt.title(f"{stem}: {col}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_{col}.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot EdgeAI-ROS latency CSV files.")
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--out-dir", default="data/results/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in args.csv:
        plot_one(Path(csv_path), out_dir)
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
