#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified generation entrypoint for HuskyMed.

Example:
    python scripts/generate.py \
        --method nbce \
        --contexts file:./data/contexts.txt \
        --task "科の医師として以下記録を要約してください" \
        --max-new-tokens 512

Contexts input formats:
    1) file:<path>  -> each line is a context snippet
    2) json:<path>  -> expects a list of strings in JSON
    3) inline comma-separated: "片段1,片段2,片段3"

Outputs:
    Prints generated text to stdout; optionally writes to --output.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root importable

from config import get_config
from model.base_loader import load_model_and_tokenizer
from model.wrappers.generation_manager import GenerationManager, NBCEStrategy, PCWStrategy
from utils import merge_short_sentences, distribution_sampled  # legacy support
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified generation script (NBCE / PCW) + legacy sweep")
    p.add_argument("--method", choices=["nbce", "pcw"], required=True, help="generation strategy")
    p.add_argument("--contexts", required=True, help="contexts source: file:<path> | json:<path> | inline")
    p.add_argument("--task", required=True, help="task instruction / prompt prefix")
    p.add_argument("--window-size", type=int, default=256, help="context window size")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--right-indentation", action="store_true", help="PCW: enable right indentation mode")
    p.add_argument("--output", type=str, default=None, help="optional output file path (single run or aggregated JSON)")
    p.add_argument("--dtype", type=str, default="auto", help="torch dtype for model loading")
    # New sweep / legacy parameters
    p.add_argument("--sampling-rates", type=str, default=None, help="comma list or range start:stop:step (e.g. 0.15:0.65:0.05) for NBCE sweep")
    p.add_argument("--merge-short", action="store_true", help="merge short sentences (legacy generate_0.1 behavior)")
    p.add_argument("--legacy-nbce", action="store_true", help="use internal legacy NBCE loop instead of wrapper strategy")
    p.add_argument("--output-prefix", type=str, default="result", help="prefix for sweep output files")
    return p.parse_args()


def load_contexts(spec: str) -> List[str]:
    if spec.startswith("file:"):
        path = spec.split(":", 1)[1]
        with open(path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    if spec.startswith("json:"):
        path = spec.split(":", 1)[1]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON contexts must be a list of strings")
        return [str(x) for x in data]
    # inline comma separated
    return [s.strip() for s in spec.split(',') if s.strip()]


def _parse_sampling_rates(spec: str | None):
    if not spec:
        return None
    spec = spec.strip()
    if ":" in spec:  # range form start:stop:step
        try:
            start, stop, step = [float(x) for x in spec.split(":")]
        except ValueError:
            raise ValueError("Invalid range format for --sampling-rates. Use start:stop:step")
        vals = []
        cur = start
        # avoid float accumulation drift by counting steps
        while cur < stop + 1e-9:
            vals.append(round(cur, 10))
            cur += step
        return vals
    # comma-separated list
    return [float(x) for x in spec.split(',') if x.strip()]


def legacy_nbce_generate(model, tokenizer, contexts_cache: list[str], task_text: str, max_new_tokens: int, window_size: int) -> str:
    """Replicates legacy generate_0.1 NBCE entropy selection + beta merge.

    NOTE: Research/debug only. Uses brute force prompt packing approach.
    """
    inputs = [task_text + "USER:" + "ASSISTANT:サマリー\n"]
    for line in contexts_cache:
        inputs.append(task_text + "USER:" + line + "ASSISTANT:サマリー\n")
    enc = tokenizer(inputs, padding='max_length', truncation=True, max_length=window_size + 1, return_tensors='pt')
    n = enc['input_ids'].shape[0]
    input_ids = enc['input_ids'].to(model.device)
    attention_mask = enc['attention_mask'].to(model.device)

    processors = []  # TopP/TopK tunable could be added later
    preds = []
    past_key_values = None
    beta = 0.65
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1]
        logits = logits - logits.logsumexp(dim=-1, keepdims=True)
        for proc in processors:
            logits = proc(input_ids, logits)
        entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
        k = entropy[1:].argmin() + 1 if logits.shape[0] > 1 else 0
        logits_max = logits[k]
        logits_uncond = logits[0]
        logits_merged = (1 + beta) * logits_max - beta * logits_uncond
        logits_final = torch.where(logits_uncond > -100, logits_merged, logits_max)
        probas = torch.nn.functional.softmax(logits_final[None], dim=-1)
        probas = torch.where(torch.isnan(probas), torch.zeros_like(probas), probas)
        next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)
        if next_tokens[0] == tokenizer.eos_token_id:
            break
        preds.append(tokenizer.decode(next_tokens[0]))
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=attention_mask.device)], dim=-1)
    return "".join(preds)


def main():
    args = parse_args()
    cfg = get_config()  # ensures directories
    model, tok, device = load_model_and_tokenizer(torch_dtype=args.dtype)

    contexts = load_contexts(args.contexts)

    # Single run (no sweep)
    sampling_rates = _parse_sampling_rates(args.sampling_rates)
    if args.method == "pcw" and sampling_rates:
        raise ValueError("--sampling-rates sweep only applies to NBCE method")

    if not sampling_rates:
        nbce = NBCEStrategy(model, tok, device, window_size=args.window_size)
        pcw = PCWStrategy(model, tok, device, window_size=args.window_size, right_indentation=args.right_indentation)
        manager = GenerationManager(nbce=nbce, pcw=pcw)
        result = manager.run(
            method=args.method,
            contexts=contexts,
            task_text=args.task,
            max_new_tokens=args.max_new_tokens,
        )
        print("=== Generation Result ===")
        print(result.text)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({
                    "method": result.strategy,
                    "task": args.task,
                    "contexts": contexts,
                    "output": result.text
                }, f, ensure_ascii=False, indent=2)
            print(f"Saved to {args.output}")
        return

    # Sweep mode for NBCE
    aggregate = {}
    for rs in sampling_rates:
        # distribution_sampled returns (cache, length_distribution)
        cache, _dist = distribution_sampled(contexts, tok, sampling_rate=rs)
        if args.merge_short:
            cache = merge_short_sentences(cache, args.window_size)
        if len(cache) <= 1:
            continue
        if args.legacy_nbce:
            text = legacy_nbce_generate(model, tok, cache, args.task, args.max_new_tokens, args.window_size)
            strategy = "legacy-nbce"
        else:
            nbce = NBCEStrategy(model, tok, device, window_size=args.window_size)
            manager = GenerationManager(nbce=nbce, pcw=None)
            res = manager.run(method="nbce", contexts=cache, task_text=args.task, max_new_tokens=args.max_new_tokens)
            text = res.text
            strategy = res.strategy
        aggregate[str(rs)] = {
            "sampling_rate": rs,
            "contexts_selected": cache,
            "task": args.task,
            "output": text,
            "strategy": strategy,
        }
        out_file = f"{args.output_prefix}_{rs}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(aggregate[str(rs)], f, ensure_ascii=False, indent=2)
        print(f"Saved intermediate sweep file: {out_file}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"sweep": aggregate}, f, ensure_ascii=False, indent=2)
        print(f"Saved aggregated sweep file: {args.output}")


if __name__ == "__main__":
    main()
