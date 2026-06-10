from __future__ import annotations

import os


def main() -> None:
    print(f"HF_TOKEN_present={bool(os.environ.get('HF_TOKEN'))}")
    print(f"HUGGING_FACE_HUB_TOKEN_present={bool(os.environ.get('HUGGING_FACE_HUB_TOKEN'))}")


if __name__ == "__main__":
    main()
