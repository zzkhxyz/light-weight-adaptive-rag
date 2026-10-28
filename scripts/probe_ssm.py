"""Experimental local Mamba cache probe; not a trained document QA compressor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def tensor_bytes(value, seen: set[int] | None = None) -> int:
    """Count tensor storage held by a model cache, avoiding repeated references."""
    import torch

    seen = seen or set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return tensor_bytes(vars(value), seen)
    return 0


def run_cached(model, input_ids, max_new_tokens: int, chunk_tokens: int) -> tuple[list[int], int, float]:
    import torch

    cache = None
    started = time.perf_counter()
    with torch.inference_mode():
        for chunk in input_ids.split(chunk_tokens, dim=1):
            output = model(input_ids=chunk, cache_params=cache, use_cache=True,
                           return_dict=True, logits_to_keep=1)
            cache = output.cache_params
        cache_size = tensor_bytes(cache)
        generated = []
        next_id = output.logits[:, -1, :].argmax(dim=-1)
        for _ in range(max_new_tokens):
            generated.append(next_id.item())
            output = model(input_ids=next_id[:, None], cache_params=cache,
                           use_cache=True, return_dict=True, logits_to_keep=1)
            cache = output.cache_params
            next_id = output.logits[:, -1, :].argmax(dim=-1)
    return generated, cache_size, round((time.perf_counter() - started) * 1000, 2)


def document(position: str, repeats: int) -> str:
    sentence = "The operations team reviews routine forms and records a weekly status update. "
    filler = sentence * repeats
    fact = "The access code for the Aurora archive is ZEBRA. "
    if position == "start":
        return fact + filler
    if position == "middle":
        return sentence * (repeats // 2) + fact + sentence * (repeats - repeats // 2)
    return filler + fact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--repeats", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/ssm-probe.json"))
    parser.add_argument("--web-output", type=Path, default=None,
                        help="Optional packaged JSON for the /lab page")
    args = parser.parse_args()
    if min(args.repeats) < 1 or args.chunk_tokens < 1 or args.max_new_tokens < 1:
        parser.error("repeats, chunk-tokens, and max-new-tokens must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).eval().to("cpu")
    cases = []
    for repeats in args.repeats:
        for position in ("start", "middle", "end"):
            prompt = document(position, repeats) + "\nQuestion: What is the access code for the Aurora archive?\nAnswer:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            ids, state_bytes, elapsed_ms = run_cached(model, input_ids, args.max_new_tokens, args.chunk_tokens)
            answer = tokenizer.decode(ids, skip_special_tokens=True).strip()
            cases.append({"tokens": input_ids.shape[1], "position": position,
                          "state_bytes": state_bytes, "latency_ms": elapsed_ms,
                          "answer": answer, "recalled": "ZEBRA" in answer.upper()})
            print(f"{input_ids.shape[1]} tokens, {position}: {answer!r}, state {state_bytes} bytes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps({"model": args.model, "method": "zero-shot cached continuation",
                         "note": "This tests recall from a pretrained recurrent cache, not trained QA or verified citations.",
                         "cases": cases}, indent=2) + "\n"
    args.output.write_text(report, encoding="utf-8")
    if args.web_output:
        args.web_output.parent.mkdir(parents=True, exist_ok=True)
        args.web_output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
