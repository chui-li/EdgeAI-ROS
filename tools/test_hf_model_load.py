from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Hugging Face causal LM loading.")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"

    print(f"model={args.model}")
    print(f"token_present={bool(token)}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch={torch.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=token,
        local_files_only=args.local_files_only,
    )
    print(f"tokenizer_ok={type(tokenizer).__name__}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=token,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,
    )
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    elif args.device == "cpu":
        model = model.to("cpu")
    print(f"model_ok={type(model).__name__}")
    print(f"device={next(model.parameters()).device}")


if __name__ == "__main__":
    main()
