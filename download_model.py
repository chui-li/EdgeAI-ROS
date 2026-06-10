from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_PRESETS = {
    "gemma": ("google/gemma-3-1b-it", "gemma-3-1b-it"),
    "llama": ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download local Hugging Face models for EdgeAI-ROS experiments."
    )
    parser.add_argument(
        "--model",
        default="gemma",
        help="Model preset or Hugging Face repo id. Presets: gemma, llama.",
    )
    parser.add_argument(
        "--local-dir",
        default="",
        help="Output directory. Defaults to the preset directory name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = MODEL_PRESETS.get(args.model.lower())

    if preset:
        repo_id, default_dir = preset
    else:
        repo_id = args.model
        default_dir = repo_id.rstrip("/").split("/")[-1]

    local_dir = Path(args.local_dir or default_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} to {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Done: {local_dir}")


if __name__ == "__main__":
    main()
