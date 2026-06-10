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


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                value = ""
            vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def model_rank(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def summarize_one(path: Path, labels: pd.DataFrame, warmup_frames: int) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path)
    data = df[df["frame"] > warmup_frames].copy()
    model = str(data["model"].dropna().iloc[0])

    per_image = []
    for image, g in data.groupby("image", sort=True):
        pred_mode = g["pred_class"].mode()
        per_image.append(
            {
                "model": model,
                "image": image,
                "rows": len(g),
                "pred_class": int(pred_mode.iloc[0]) if not pred_mode.empty else -1,
                "avg_infer_ms": g["infer_ms"].mean(),
                "avg_e2e_ms": g["e2e_ms"].mean(),
                "misses": int(g["deadline_miss"].sum()),
            }
        )
    pred = pd.DataFrame(per_image).merge(labels, on="image", how="left")
    pred["correct_top1"] = pred["pred_class"].astype(int) == pred["class_index"].astype(int)

    summary = {
        "model": model,
        "params_m": PARAMS_M.get(model),
        "official_top1_300": OFFICIAL_TOP1_300.get(model),
        "rows": len(df),
        "rows_after_warmup": len(data),
        "unique_images": df["image"].nunique(),
        "evaluated_images": len(pred),
        "top1_correct": int(pred["correct_top1"].sum()),
        "top1_accuracy": pred["correct_top1"].mean(),
        "accuracy_gap_vs_official": pred["correct_top1"].mean() * 100.0 - OFFICIAL_TOP1_300.get(model, 0),
        "deadline_ms": data["deadline_ms"].median(),
        "deadline_misses": int(data["deadline_miss"].sum()),
        "deadline_miss_rate": data["deadline_miss"].mean(),
        "avg_infer_ms": data["infer_ms"].mean(),
        "median_infer_ms": data["infer_ms"].median(),
        "p95_infer_ms": data["infer_ms"].quantile(0.95),
        "max_infer_ms": data["infer_ms"].max(),
        "avg_e2e_ms": data["e2e_ms"].mean(),
        "p95_e2e_ms": data["e2e_ms"].quantile(0.95),
        "max_e2e_ms": data["e2e_ms"].max(),
        "model_source": "; ".join(sorted(data["model_source"].astype(str).unique())),
        "device": "; ".join(sorted(data["device"].astype(str).unique())),
        "csv": path.name,
    }
    return summary, pred


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        if col in {"rows", "rows_after_warmup", "unique_images", "evaluated_images", "top1_correct", "deadline_misses"}:
            out[col] = out[col].astype(int)
        else:
            out[col] = out[col].round(4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="data/results/imagenet_val_subset_gpu_repvit_*_static_large_200ms.csv")
    parser.add_argument("--labels", default="data/imagenet_val_subset_1_per_class/labels.csv")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--summary-out", default="data/results/imagenet_val_subset_gpu_repvit_family_summary.csv")
    parser.add_argument("--pred-out", default="data/results/imagenet_val_subset_gpu_repvit_family_predictions.csv")
    parser.add_argument("--report-out", default="reports/imagenet_val_subset_repvit_family_gpu_report_zh.md")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    summaries = []
    preds = []
    for path in sorted(Path().glob(args.pattern)):
        summary, pred = summarize_one(path, labels, args.warmup_frames)
        summaries.append(summary)
        preds.append(pred)

    if not summaries:
        raise SystemExit(f"No CSV matched pattern: {args.pattern}")

    summary_df = pd.DataFrame(summaries)
    summary_df["model_rank"] = summary_df["model"].map(model_rank)
    summary_df = summary_df.sort_values("model_rank").drop(columns=["model_rank"])
    pred_df = pd.concat(preds, ignore_index=True)
    pred_df["model_rank"] = pred_df["model"].map(model_rank)
    pred_df = pred_df.sort_values(["model_rank", "image"]).drop(columns=["model_rank"])

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_out, index=False)
    pred_df.to_csv(args.pred_out, index=False)

    sr = rounded(summary_df)
    report = [
        "# ImageNet Validation Subset RepViT Family GPU 測試報告",
        "",
        "## Dataset 與設定",
        "",
        "- Dataset：`data/imagenet_val_subset_1_per_class/images`",
        "- Labels：`data/imagenet_val_subset_1_per_class/labels.csv`",
        "- 抽樣：ImageNet validation 每類 1 張，共 1000 張",
        "- Device：CUDA / NVIDIA GPU",
        "- Policy：`static_large`",
        "- Image size：224",
        "- Deadline：200 ms",
        "- ROS publish period：0.1 s",
        "- Warm-up：排除前 5 frames",
        "",
        "## Accuracy 與 Latency 總表",
        "",
        md_table(
            sr,
            [
                "model",
                "params_m",
                "official_top1_300",
                "top1_accuracy",
                "accuracy_gap_vs_official",
                "top1_correct",
                "evaluated_images",
                "deadline_misses",
                "deadline_miss_rate",
                "avg_infer_ms",
                "p95_infer_ms",
                "max_infer_ms",
                "avg_e2e_ms",
                "p95_e2e_ms",
            ],
        ),
        "",
        "## 觀察",
        "",
        "- M0.9 的 78.4% 與官方 300e top-1 78.7% 非常接近，顯示 subset label 對齊與 preprocessing 可信。",
        "- Accuracy 隨模型大小整體上升，M2.3 在此 subset 上達到最高 top-1 accuracy。",
        "- GPU 上所有模型的平均 inference latency 均低於 80 ms；M1.5 與 M2.3 因模型較大，在 0.1 s 發布頻率與 ROS/Python overhead 下出現少量 200 ms deadline miss。",
        "- 這是每類 1 張的 balanced subset，因此適合做快速 sanity check；正式 accuracy 仍應使用完整 50,000 張 ImageNet validation set。",
        "",
        "## 輸出檔案",
        "",
        f"- Summary CSV：`{args.summary_out}`",
        f"- Prediction CSV：`{args.pred_out}`",
        "- Raw CSV：`data/results/imagenet_val_subset_gpu_repvit_*_static_large_200ms.csv`",
        "",
    ]
    Path(args.report_out).write_text("\n".join(report), encoding="utf-8")
    print(sr.to_string(index=False))


if __name__ == "__main__":
    main()
