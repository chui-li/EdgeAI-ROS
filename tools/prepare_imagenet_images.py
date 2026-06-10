from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def resize_keep_aspect(img, max_side: int):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="Resize real-image inputs for ROS image publishing.")
    parser.add_argument("--src", default="data/imagenet_test_images")
    parser.add_argument("--dst", default="data/imagenet_test_images_resized")
    parser.add_argument("--max-side", type=int, default=512)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for path in sorted(src.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip unreadable: {path}")
            continue
        resized = resize_keep_aspect(img, args.max_side)
        out = dst / path.name
        cv2.imwrite(str(out), resized)
        print(f"{path.name}: {img.shape[1]}x{img.shape[0]} -> {resized.shape[1]}x{resized.shape[0]}")


if __name__ == "__main__":
    main()
