#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified evaluation script to build performance matrices.

Input file can be JSON (list or {items:[...]}) or CSV with columns including:
 - prediction/output, reference/ref/smr_human
 - optional grouping fields: method, sampling_rate, window_size, run, etc.

Example:
  python scripts/evaluate.py \
    --file ./data/preds.json \
    --metrics rouge1,rougeL \
    --group-by method,sampling_rate,window_size \
    --out-json ./out/matrix.json \
    --out-csv ./out/matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List

from evaluation import (
    EvaluationEngine,
    Metric,
    RougeMetric,
    SacreBleuMetric,
    parse_rouge_variants,
    write_csv,
    write_json,
)


def _load_any(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "items" in data:
            return list(data["items"])  # assume list of dicts
        if isinstance(data, list):
            return data
        raise ValueError("Unsupported JSON structure: expect list or {items: [...]} ")
    elif ext in (".csv",):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predictions and build performance matrix")
    p.add_argument("--file", required=True, help="input predictions file (json/csv)")
    p.add_argument("--metrics", type=str, default="rouge1,rougeL")
    p.add_argument("--group-by", type=str, default="method,sampling_rate,window_size")
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--out-csv", type=str, default=None)
    return p.parse_args()
def _build_metrics(names: Iterable[str]) -> List[Metric]:
    requested = [name.strip() for name in names if name.strip()]
    rouge_variants = parse_rouge_variants(requested)
    metrics: List[Metric] = []
    if rouge_variants:
        metrics.append(RougeMetric(rouge_variants))
    if "bleu" in requested:
        metrics.append(SacreBleuMetric())
    unsupported = sorted(set(requested) - set(rouge_variants) - {"bleu"})
    if unsupported:
        raise ValueError(f"Unsupported metrics requested: {', '.join(unsupported)}")
    return metrics


def main():
    args = parse_args()
    rows = _load_any(args.file)
    metric_names = [m.strip() for m in args.metrics.split(',') if m.strip()]
    engine = EvaluationEngine(metrics=_build_metrics(metric_names))
    enriched = engine.score_rows(rows)

    # Collect available metric scalar keys (ROUGE precision/recall/f-measure, BLEU score, ...)
    metric_keys = sorted(
        {
            key
            for row in enriched
            for key, value in row.items()
            if key.endswith(("_f", "_p", "_r")) or key in {"bleu"}
        }
    )
    group_keys = [g.strip() for g in args.group_by.split(',') if g.strip()]

    matrix = engine.aggregate(enriched, group_keys=group_keys, metric_keys=metric_keys)

    if args.out_json:
        write_json(args.out_json, {"rows": matrix})
        print(f"Wrote JSON matrix to: {args.out_json}")
    if args.out_csv:
        write_csv(args.out_csv, matrix)
        print(f"Wrote CSV matrix to: {args.out_csv}")
    if not args.out_json and not args.out_csv:
        # print to stdout succinctly
        preview = matrix[:5]
        print(json.dumps({"rows": preview, "total": len(matrix)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
