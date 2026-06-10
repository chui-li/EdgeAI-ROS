from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_ORDER = [
    "repvit_m0_6",
    "repvit_m0_9",
    "repvit_m1_0",
    "repvit_m1_1",
    "repvit_m1_5",
    "repvit_m2_3",
]

PARAMS_M = {
    "repvit_m0_6": 2.487,
    "repvit_m0_9": 5.104,
    "repvit_m1_0": 6.853,
    "repvit_m1_1": 8.289,
    "repvit_m1_5": 14.130,
    "repvit_m2_3": 23.047,
}

OFFICIAL_TOP1_300 = {
    "repvit_m0_6": 74.1,
    "repvit_m0_9": 78.7,
    "repvit_m1_0": 80.0,
    "repvit_m1_1": 80.7,
    "repvit_m1_5": 82.3,
    "repvit_m2_3": 83.3,
}

OFFICIAL_MACS_G = {
    "repvit_m0_6": None,
    "repvit_m0_9": 0.8,
    "repvit_m1_0": 1.1,
    "repvit_m1_1": 1.3,
    "repvit_m1_5": 2.3,
    "repvit_m2_3": 4.5,
}


def imagenet_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def model_sort_key(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def read_runs(pattern: str, labels: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file in sorted(Path().glob(pattern)):
        df = pd.read_csv(file)
        if df.empty:
            continue
        device = "gpu" if "model_benchmark_gpu_" in file.name else "cpu"
        df["run_file"] = file.name
        df["bench_device"] = device
        if labels:
            df["pred_label"] = df["pred_class"].astype(int).map(
                lambda i: labels[i] if 0 <= i < len(labels) else "unknown"
            )
        else:
            df["pred_label"] = df["pred_class"].astype(str)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No data matched pattern: {pattern}")
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, warmup_frames: int) -> pd.DataFrame:
    data = df[df["frame"] > warmup_frames].copy()
    rows = []
    for (device, model), g in data.groupby(["bench_device", "model"], sort=False):
        miss = int(g["deadline_miss"].sum())
        rows.append(
            {
                "device": device.upper(),
                "model": model,
                "params_m": PARAMS_M.get(model),
                "official_top1_300": OFFICIAL_TOP1_300.get(model),
                "official_macs_g": OFFICIAL_MACS_G.get(model),
                "rows": len(g),
                "deadline_ms": g["deadline_ms"].median(),
                "deadline_miss": miss,
                "deadline_miss_rate": miss / len(g) if len(g) else 0,
                "avg_infer_ms": g["infer_ms"].mean(),
                "median_infer_ms": g["infer_ms"].median(),
                "p95_infer_ms": g["infer_ms"].quantile(0.95),
                "max_infer_ms": g["infer_ms"].max(),
                "avg_e2e_ms": g["e2e_ms"].mean(),
                "p95_e2e_ms": g["e2e_ms"].quantile(0.95),
                "max_e2e_ms": g["e2e_ms"].max(),
                "avg_cpu_percent": g["cpu_percent"].mean(),
                "avg_rss_mb": g["process_rss_mb"].mean(),
                "model_source": "; ".join(sorted(g["model_source"].astype(str).unique())),
                "runtime_device": "; ".join(sorted(g["device"].astype(str).unique())),
            }
        )
    out = pd.DataFrame(rows)
    out["model_rank"] = out["model"].map(model_sort_key)
    return out.sort_values(["device", "model_rank"]).drop(columns=["model_rank"])


def speedup(summary: pd.DataFrame) -> pd.DataFrame:
    cpu = summary[summary["device"] == "CPU"].set_index("model")
    gpu = summary[summary["device"] == "GPU"].set_index("model")
    rows = []
    for model in MODEL_ORDER:
        if model not in cpu.index or model not in gpu.index:
            continue
        rows.append(
            {
                "model": model,
                "cpu_avg_infer_ms": cpu.loc[model, "avg_infer_ms"],
                "gpu_avg_infer_ms": gpu.loc[model, "avg_infer_ms"],
                "speedup_x": cpu.loc[model, "avg_infer_ms"] / gpu.loc[model, "avg_infer_ms"],
                "cpu_miss_rate": cpu.loc[model, "deadline_miss_rate"],
                "gpu_miss_rate": gpu.loc[model, "deadline_miss_rate"],
            }
        )
    return pd.DataFrame(rows)


def prediction_summary(df: pd.DataFrame, warmup_frames: int) -> pd.DataFrame:
    data = df[df["frame"] > warmup_frames].copy()
    rows = []
    for (device, model, image), g in data.groupby(["bench_device", "model", "image"], sort=False):
        mode_label = g["pred_label"].mode()
        mode_class = g["pred_class"].mode()
        label = mode_label.iloc[0] if not mode_label.empty else ""
        pred_class = int(mode_class.iloc[0]) if not mode_class.empty else -1
        rows.append(
            {
                "device": device.upper(),
                "model": model,
                "image": image,
                "rows": len(g),
                "mode_pred_class": pred_class,
                "mode_pred_label": label,
                "unique_pred_labels": " | ".join(sorted(g["pred_label"].astype(str).unique())),
                "avg_infer_ms": g["infer_ms"].mean(),
                "misses": int(g["deadline_miss"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["model_rank"] = out["model"].map(model_sort_key)
    return out.sort_values(["device", "model_rank", "image"]).drop(columns=["model_rank"])


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in view.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                value = ""
            values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *rows])


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        if col in {"rows", "deadline_miss", "mode_pred_class", "misses"}:
            out[col] = out[col].astype(int)
        else:
            out[col] = out[col].round(3)
    return out


def write_report(summary: pd.DataFrame, speed: pd.DataFrame, pred: pd.DataFrame, path: Path) -> None:
    summary_r = rounded(summary)
    speed_r = rounded(speed)
    pred_r = rounded(pred)

    cpu = summary_r[summary_r["device"] == "CPU"]
    gpu = summary_r[summary_r["device"] == "GPU"]

    lines = [
        "# RepViT Model Family Benchmark Report",
        "",
        "## 實驗設定",
        "",
        "- 測試模型：RepViT-M0.6、M0.9、M1.0、M1.1、M1.5、M2.3。",
        "- 權重：THU-MIG RepViT official `distill_300e` checkpoint。",
        "- 資料集：`data/imagenet_labeled_images_resized`，共 7 張真實 ImageNet 類型照片。",
        "- Pipeline：ROS2 `image_publisher -> adaptive_repvit_node -> edgeai_logger`。",
        "- Policy：`static_large`，三層模型皆固定為同一 variant，解析度皆為 `224x224`。",
        "- Deadline：`200 ms`。",
        "- 裝置：CPU 與 NVIDIA GeForce RTX 3060 Laptop GPU。",
        "- 統計方式：每組執行 90 秒，排除前 5 frames 作為 warm-up。",
        "",
        "## CPU 結果總表",
        "",
        md_table(
            cpu,
            [
                "model",
                "params_m",
                "official_top1_300",
                "official_macs_g",
                "rows",
                "deadline_miss",
                "deadline_miss_rate",
                "avg_infer_ms",
                "median_infer_ms",
                "p95_infer_ms",
                "max_infer_ms",
                "avg_e2e_ms",
            ],
        ),
        "",
        "## GPU 結果總表",
        "",
        md_table(
            gpu,
            [
                "model",
                "params_m",
                "official_top1_300",
                "official_macs_g",
                "rows",
                "deadline_miss",
                "deadline_miss_rate",
                "avg_infer_ms",
                "median_infer_ms",
                "p95_infer_ms",
                "max_infer_ms",
                "avg_e2e_ms",
            ],
        ),
        "",
        "## CPU/GPU 加速比",
        "",
        md_table(speed_r, ["model", "cpu_avg_infer_ms", "gpu_avg_infer_ms", "speedup_x", "cpu_miss_rate", "gpu_miss_rate"]),
        "",
        "## 分類結果完整表",
        "",
        md_table(
            pred_r,
            [
                "device",
                "model",
                "image",
                "rows",
                "mode_pred_class",
                "mode_pred_label",
                "unique_pred_labels",
                "avg_infer_ms",
                "misses",
            ],
        ),
        "",
        "## 觀察",
        "",
        "- 在 CPU 上，M0.6 到 M1.1 大致可維持 200 ms deadline；M1.5 開始出現較明顯 deadline miss，M2.3 在 CPU 上已不適合此 deadline。",
        "- 在 GPU 上，所有模型平均 inference latency 都明顯低於 CPU；M2.3 仍有單次 deadline miss，主要代表 ROS/Python pipeline 中偶發端到端 jitter 仍需留意。",
        "- GPU 對大模型的收益更明顯，M2.3 的 CPU/GPU 平均 inference 加速比最高。",
        "- 分類結果在大部分影像上穩定；少數模型與裝置對 `cat.jpg`、`dog.jpg` 會在相近 ImageNet 類別間切換，這是 top-1 label granularity 的典型現象。",
        "",
        "## 檔案輸出",
        "",
        "- Summary CSV：`data/results/repvit_model_family_summary.csv`",
        "- Speedup CSV：`data/results/repvit_model_family_speedup.csv`",
        "- Prediction CSV：`data/results/repvit_model_family_predictions.csv`",
        "- Raw CSV：`data/results/model_benchmark_{cpu,gpu}_repvit_*_200ms.csv`",
        "- Raw log：`data/results/model_benchmark_{cpu,gpu}_repvit_*_200ms.log`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="data/results/model_benchmark_*_200ms.csv")
    parser.add_argument("--labels", default="data/imagenet_classes.txt")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--summary-out", default="data/results/repvit_model_family_summary.csv")
    parser.add_argument("--speedup-out", default="data/results/repvit_model_family_speedup.csv")
    parser.add_argument("--pred-out", default="data/results/repvit_model_family_predictions.csv")
    parser.add_argument("--report-out", default="reports/repvit_model_family_benchmark_zh.md")
    args = parser.parse_args()

    labels = imagenet_labels(Path(args.labels))
    df = read_runs(args.pattern, labels)
    summary = summarize(df, args.warmup_frames)
    speed = speedup(summary)
    pred = prediction_summary(df, args.warmup_frames)

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    speed.to_csv(args.speedup_out, index=False)
    pred.to_csv(args.pred_out, index=False)
    write_report(summary, speed, pred, Path(args.report_out))

    print(rounded(summary).to_string(index=False))
    print()
    print(rounded(speed).to_string(index=False))


if __name__ == "__main__":
    main()
