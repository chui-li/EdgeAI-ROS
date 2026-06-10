from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _png_chunk(tag: bytes, data: bytes):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, pixels: bytes):
    raw_rows = []
    stride = width * 3
    for y in range(height):
        raw_rows.append(b"\x00" + pixels[y * stride : (y + 1) * stride])
    raw = b"".join(raw_rows)
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = header
    data += _png_chunk(b"IHDR", ihdr)
    data += _png_chunk(b"IDAT", zlib.compress(raw, level=6))
    data += _png_chunk(b"IEND", b"")
    path.write_bytes(data)


def make_pixels(width: int, height: int, idx: int):
    cx = 120 + idx * 160
    cy = height // 2
    radius = 64
    out = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * 3
            r = int(220 - 160 * y / max(1, height - 1))
            g = int(20 + 200 * x / max(1, width - 1))
            b = 30 + idx * 30
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                r, g, b = 255, 220, 80
            out[offset : offset + 3] = bytes((r, g, b))
    return bytes(out)


def main():
    out_dir = Path("data/test_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        width, height = 640, 480
        write_png(out_dir / f"test_{idx}.png", width, height, make_pixels(width, height, idx))
    print(f"Wrote test images to {out_dir}")


if __name__ == "__main__":
    main()
