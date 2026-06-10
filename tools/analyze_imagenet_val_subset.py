from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/results/imagenet_val_subset_gpu_repvit_m0_9_static_large_200ms.csv")
    parser.add_argument("--labels", default="data/imagenet_val_subset_1_per_class/labels.csv")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--summary-out", default="data/results/imagenet_val_subset_gpu_repvit_m0_9_summary.csv")
    parser.add_argument("--pred-out", default="data/results/imagenet_val_subset_gpu_repvit_m0_9_predictions.csv")
    parser.add_argument("--report-out", default="reports/imagenet_val_subset_repvit_m0_9_gpu_report_zh.md")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    labels = pd.read_csv(args.labels)
    data = df[df["frame"] > args.warmup_frames].copy()

    grouped = []
    for image, g in data.groupby("image", sort=True):
        pred_mode = g["pred_class"].mode()
        grouped.append(
            {
                "image": image,
                "rows": len(g),
                "pred_class": int(pred_mode.iloc[0]) if not pred_mode.empty else -1,
                "avg_infer_ms": g["infer_ms"].mean(),
                "avg_e2e_ms": g["e2e_ms"].mean(),
                "misses": int(g["deadline_miss"].sum()),
            }
        )
    pred = pd.DataFrame(grouped).merge(labels, on="image", how="left")
    pred["correct_top1"] = pred["pred_class"].astype(int) == pred["class_index"].astype(int)

    summary = pd.DataFrame(
        [
            {
                "csv": Path(args.csv).name,
                "rows": len(df),
                "rows_after_warmup": len(data),
                "unique_images": df["image"].nunique(),
                "unique_images_after_warmup": data["image"].nunique(),
                "evaluated_images": len(pred),
                "top1_correct": int(pred["correct_top1"].sum()),
                "top1_accuracy": pred["correct_top1"].mean(),
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
                "model": "; ".join(sorted(data["model"].astype(str).unique())),
                "device": "; ".join(sorted(data["device"].astype(str).unique())),
                "model_source": "; ".join(sorted(data["model_source"].astype(str).unique())),
            }
        ]
    )

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    pred.to_csv(args.pred_out, index=False)

    s = summary.copy()
    for col in s.select_dtypes(include="number").columns:
        s[col] = s[col].round(4)

    wrong = pred[~pred["correct_top1"]].head(25).copy()
    wrong["avg_infer_ms"] = wrong["avg_infer_ms"].round(3)

    report = [
        "# ImageNet Validation Subset RepViT-M0.9 GPU 測試報告",
        "",
        "## Dataset",
        "",
        "- 來源：`imagenet-object-localization-challenge.zip`",
        "- Split：ILSVRC validation",
        "- 抽樣方式：每個 synset/class 隨機抽 1 張，seed = 2026",
        "- 圖片數：1000 張，覆蓋 ImageNet 1000 類",
        "- 圖片資料夾：`data/imagenet_val_subset_1_per_class/images`",
        "- 標籤檔：`data/imagenet_val_subset_1_per_class/labels.csv`",
        "",
        "## 測試設定",
        "",
        "- Model：RepViT-M0.9 official `distill_300e` checkpoint",
        "- Device：CUDA / NVIDIA GPU",
        "- Policy：`static_large`",
        "- Image size：224",
        "- Deadline：200 ms",
        "- ROS publish period：0.1 s",
        "- Warm-up：排除前 5 frames",
        "",
        "## Summary",
        "",
        md_table(
            s,
            [
                "rows_after_warmup",
                "evaluated_images",
                "top1_correct",
                "top1_accuracy",
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
        "## 前 25 筆錯誤分類範例",
        "",
        md_table(
            wrong,
            [
                "image",
                "synset",
                "class_index",
                "label",
                "pred_class",
                "rows",
                "avg_infer_ms",
            ],
        ),
        "",
        "## 輸出檔案",
        "",
        f"- Summary CSV：`{args.summary_out}`",
        f"- Prediction CSV：`{args.pred_out}`",
        f"- Raw CSV：`{args.csv}`",
        "",
    ]
    Path(args.report_out).write_text("\n".join(report), encoding="utf-8")

    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
