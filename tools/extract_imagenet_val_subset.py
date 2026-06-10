from __future__ import annotations

import argparse
import csv
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def load_synset_mapping(zipf: zipfile.ZipFile) -> dict[str, str]:
    with zipf.open("LOC_synset_mapping.txt") as f:
        lines = f.read().decode("utf-8").splitlines()
    mapping = {}
    for line in lines:
        if not line.strip():
            continue
        synset, label = line.split(" ", 1)
        mapping[synset] = label
    return mapping


def load_synset_indices(zipf: zipfile.ZipFile) -> dict[str, int]:
    with zipf.open("LOC_synset_mapping.txt") as f:
        lines = f.read().decode("utf-8").splitlines()
    return {line.split(" ", 1)[0]: idx for idx, line in enumerate(lines) if line.strip()}


def load_val_solution(zipf: zipfile.ZipFile) -> dict[str, str]:
    with zipf.open("LOC_val_solution.csv") as f:
        text = f.read().decode("utf-8").splitlines()
    rows = csv.DictReader(text)
    labels = {}
    for row in rows:
        # PredictionString can contain multiple boxes/classes. The first token is
        # sufficient for classification-label sampling.
        labels[row["ImageId"]] = row["PredictionString"].split()[0]
    return labels


def resize_keep_aspect(img, max_side: int):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def decode_jpeg(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a balanced ImageNet validation subset from Kaggle CLS-LOC zip.")
    parser.add_argument("--zip", default="imagenet-object-localization-challenge.zip")
    parser.add_argument("--out", default="data/imagenet_val_subset_1_per_class")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    with zipfile.ZipFile(zip_path) as zipf:
        synset_names = load_synset_mapping(zipf)
        synset_indices = load_synset_indices(zipf)
        val_labels = load_val_solution(zipf)

        by_synset: dict[str, list[str]] = defaultdict(list)
        for image_id, synset in val_labels.items():
            by_synset[synset].append(image_id)

        selected: list[tuple[str, str]] = []
        for synset in sorted(by_synset):
            candidates = sorted(by_synset[synset])
            random.shuffle(candidates)
            for image_id in sorted(candidates[: args.per_class]):
                selected.append((image_id, synset))

        labels_path = out_dir / "labels.csv"
        with labels_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "image_id",
                    "synset",
                    "class_index",
                    "label",
                    "source_zip_path",
                    "width",
                    "height",
                ],
            )
            writer.writeheader()

            for idx, (image_id, synset) in enumerate(selected, start=1):
                src = f"ILSVRC/Data/CLS-LOC/val/{image_id}.JPEG"
                out_name = f"{image_id}_{synset}.jpg"
                out_path = image_dir / out_name

                if args.overwrite or not out_path.exists():
                    data = zipf.read(src)
                    img = decode_jpeg(data)
                    if img is None:
                        raise RuntimeError(f"Failed to decode {src}")
                    img = resize_keep_aspect(img, args.max_side)
                    cv2.imwrite(str(out_path), img)
                else:
                    img = cv2.imread(str(out_path))
                    if img is None:
                        raise RuntimeError(f"Failed to read existing {out_path}")

                h, w = img.shape[:2]
                writer.writerow(
                    {
                        "image": out_name,
                        "image_id": image_id,
                        "synset": synset,
                        "class_index": synset_indices.get(synset, -1),
                        "label": synset_names.get(synset, ""),
                        "source_zip_path": src,
                        "width": w,
                        "height": h,
                    }
                )

                if idx % 100 == 0:
                    print(f"extracted {idx}/{len(selected)}")

    print(f"done: {len(selected)} images")
    print(f"images: {image_dir}")
    print(f"labels: {labels_path}")


if __name__ == "__main__":
    main()
