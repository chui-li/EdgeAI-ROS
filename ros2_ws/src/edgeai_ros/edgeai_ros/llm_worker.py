from __future__ import annotations

import os
import time
from typing import Tuple

import torch


class FallbackLLM:
    def __init__(self):
        self.words = (
            "fallback lightweight generator used for operating system aware "
            "inference runtime evaluation under resource constraints "
        ).split()

    def generate_stream(self, prompt: str, max_new_tokens: int):
        for i in range(max(1, int(max_new_tokens))):
            time.sleep(0.006)
            yield self.words[i % len(self.words)]


def load_llm(model_name: str, device: torch.device, local_files_only: bool = False):
    """Load a Hugging Face causal LM safely.

    If transformers/model loading fails, return FallbackLLM so OS-level
    scheduling experiments can still run.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        dtype_name = os.environ.get("LLM_TORCH_DTYPE", "auto").lower()
        torch_dtype = "auto"
        if dtype_name in {"float16", "fp16", "half"}:
            torch_dtype = torch.float16
        elif dtype_name in {"bfloat16", "bf16"}:
            torch_dtype = torch.bfloat16
        elif dtype_name in {"float32", "fp32"}:
            torch_dtype = torch.float32
        elif device.type == "cuda":
            torch_dtype = torch.float16

        load_kwargs = {
            "local_files_only": local_files_only,
            "token": token,
        }
        model_kwargs = {
            **load_kwargs,
            "torch_dtype": torch_dtype,
        }

        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval().to(device)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[INFO] Loaded LLM: {model_name}")
        return model, tokenizer, False

    except Exception as e:
        print(f"[WARN] LLM load failed for {model_name}: {e}")
        print("[WARN] Use FallbackLLM. OS scheduling metrics remain valid; language quality is not evaluated.")
        return FallbackLLM(), None, True


def compress_prompt(prompt: str, max_words: int):
    """Simple prompt compression proxy.

    Keeps the beginning and ending words when the prompt is longer than budget.
    """
    words = prompt.split()
    max_words = max(8, int(max_words))
    if len(words) <= max_words:
        return prompt

    head = max_words // 3
    tail = max_words - head
    return " ".join(words[:head] + ["..."] + words[-tail:])


def get_model_context_limit(model, tokenizer=None) -> int:
    """Return a safe maximum context length for a causal LM.

    GPT-2-like models have fixed position embeddings. If the prompt length or
    requested context length exceeds that limit, model.generate() can crash with:

        IndexError: index out of range in self

    This helper checks common config names and tokenizer limits.
    """
    candidates = []

    cfg = getattr(model, "config", None)
    if cfg is not None:
        for name in [
            "max_position_embeddings",
            "n_positions",
            "n_ctx",
            "seq_length",
            "max_sequence_length",
        ]:
            value = getattr(cfg, name, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)

    if tokenizer is not None:
        tok_max = getattr(tokenizer, "model_max_length", None)
        # Some tokenizers use a giant sentinel value, ignore it.
        if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
            candidates.append(tok_max)

    if not candidates:
        return 2048

    return int(min(candidates))


@torch.no_grad()
def generate_once(
    model,
    tokenizer,
    fallback: bool,
    prompt: str,
    context_length: int,
    max_new_tokens: int,
    device: torch.device,
):
    """Run one LLM generation request safely.

    Important fix:
    The runtime may choose context_length=2048 for the large LLM config, but
    GPT-2 / tiny-gpt2 only supports 1024 positions. If tokenized prompt length
    or prompt + generated tokens exceeds the model's position embedding size,
    Transformers raises an IndexError. We clamp both input context and
    max_new_tokens to the model's actual maximum sequence length.
    """
    context_length = max(8, int(context_length))
    max_new_tokens = max(1, int(max_new_tokens))

    if fallback:
        # Use prompt compression in fallback mode too, so the CSV still records
        # a meaningful prompt budget effect.
        prompt_words_budget = max(8, context_length // 2)
        used_prompt = compress_prompt(prompt, prompt_words_budget)

        t0 = time.perf_counter()
        first_time = None
        count = 0
        for _ in model.generate_stream(used_prompt, max_new_tokens):
            count += 1
            if first_time is None:
                first_time = time.perf_counter()

        total_ms = (time.perf_counter() - t0) * 1000.0
        ttft_ms = ((first_time - t0) * 1000.0) if first_time is not None else total_ms
        tpot_ms = total_ms / max(1, count)
        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "tokens_per_sec": count / max(1e-9, total_ms / 1000.0),
            "output_tokens": count,
            "prompt_words": len(used_prompt.split()),
        }

    model_limit = get_model_context_limit(model, tokenizer)

    # Reserve room for generated tokens. If the requested output is too large,
    # shrink it first. Keep at least one generation token.
    safe_max_new_tokens = min(max_new_tokens, max(1, model_limit - 1))

    # Input tokens must leave enough room for generated tokens.
    safe_input_len = max(1, min(context_length, model_limit - safe_max_new_tokens))

    # If safe_input_len is too small because max_new_tokens is huge, reduce
    # max_new_tokens and allow a reasonable input length.
    if safe_input_len < 8 and model_limit > 16:
        safe_max_new_tokens = max(1, min(max_new_tokens, model_limit // 4))
        safe_input_len = max(8, min(context_length, model_limit - safe_max_new_tokens))

    # Word-level prompt compression before tokenization keeps tokenization fast
    # for extremely repeated prompts.
    prompt_words_budget = max(8, safe_input_len // 2)
    used_prompt = compress_prompt(prompt, prompt_words_budget)

    inputs = tokenizer(
        used_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=safe_input_len,
    ).to(device)

    input_len = int(inputs["input_ids"].shape[1])

    # Final guard: generated sequence length must never exceed position limit.
    remaining = max(1, model_limit - input_len)
    safe_max_new_tokens = max(1, min(safe_max_new_tokens, remaining))

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=safe_max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_ms = (time.perf_counter() - t0) * 1000.0

    output_len = int(output.shape[1])
    new_tokens = max(1, output_len - input_len)

    # For non-streaming generate(), true TTFT is not exposed. Approximate TTFT
    # using per-token generation time.
    tpot_ms = total_ms / max(1, new_tokens)
    ttft_ms = tpot_ms
    tokens_per_sec = new_tokens / max(1e-9, total_ms / 1000.0)

    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tokens_per_sec": tokens_per_sec,
        "output_tokens": new_tokens,
        "prompt_words": len(used_prompt.split()),
    }
