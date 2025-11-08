from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, load_dataset

__all__ = (
    "PipelineConfig",
    "InstructionRecord",
    "build_instruction_records",
    "prepare_instruction_dataset",
    "write_instruction_chunks",
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for transforming raw hospital notes into instruction records."""

    smr_field: str = "smrs"
    smr_unit_field: str = "smr"
    smr_date_key: str = "INPUT_DATE"
    smr_type_key: str = "EMR_TYPE"
    smr_section_key: str = "SECTION"
    record_field: str = "records"
    record_items_key: str = "emr"
    record_section_key: str = "section"
    record_timestamp_key: str = "ts"
    item_type_keys: Tuple[str, ...] = ("data_type", "EMR_DATA_TYPE", "type")
    item_text_keys: Tuple[str, ...] = ("text", "EMR_TEXT", "value")
    fallback_target_types: Dict[str, str] = field(
        default_factory=lambda: {"S": "治療経過", "O": "治療経過", "A": "治療経過", "P": "治療経過"}
    )


@dataclass
class InstructionRecord:
    section: str
    smr_type: str
    merged: List[Tuple[str, str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {"section": self.section, "smr_type": self.smr_type, "merged": self.merged}


def _normalise_item(raw: Dict[str, Any], cfg: PipelineConfig) -> Optional[Tuple[str, str]]:
    for key in cfg.item_type_keys:
        data_type = raw.get(key)
        if data_type:
            break
    else:
        return None

    for key in cfg.item_text_keys:
        text = raw.get(key)
        if text:
            break
    else:
        return None

    return str(data_type), str(text)


def _parse_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    text = str(value)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d")
        except ValueError:
            pass
    return None


def _collect_context_before(
    records: Sequence[Dict[str, Any]],
    cutoff: Optional[datetime],
    cfg: PipelineConfig,
) -> List[Tuple[str, str]]:
    collected: List[Tuple[str, str]] = []
    if cutoff is None:
        return collected

    for item in records or []:
        ts = _parse_date(item.get(cfg.record_timestamp_key))
        if ts is None or ts > cutoff:
            continue
        for raw_note in item.get(cfg.record_items_key) or []:
            norm = _normalise_item(raw_note, cfg)
            if norm:
                collected.append(norm)
    return collected


def _collect_target_items(raw: Dict[str, Any], cfg: PipelineConfig) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for note in raw.get(cfg.smr_unit_field) or []:
        norm = _normalise_item(note, cfg)
        if norm:
            entries.append(norm)
    return entries


def _merge_entries(
    contexts: Sequence[Tuple[str, str]],
    targets: Sequence[Tuple[str, str]],
    cfg: PipelineConfig,
) -> List[Tuple[str, str, str]]:
    if not contexts or not targets:
        return []

    context_by_type: Dict[str, set] = defaultdict(set)
    target_by_type: Dict[str, set] = defaultdict(set)
    for data_type, text in contexts:
        context_by_type[data_type].add(text)
    for data_type, text in targets:
        target_by_type[data_type].add(text)

    merged: List[Tuple[str, str, str]] = []
    for context_type, context_texts in context_by_type.items():
        target_type = context_type if context_type in target_by_type else cfg.fallback_target_types.get(context_type)
        if target_type and target_type in target_by_type:
            for context_text in context_texts:
                for target_text in target_by_type[target_type]:
                    merged.append((target_type, context_text, target_text))
    return merged


def _build_records_from_patient(example: Dict[str, Any], cfg: PipelineConfig) -> List[InstructionRecord]:
    records = example.get(cfg.record_field) or []
    results: List[InstructionRecord] = []

    for smr in example.get(cfg.smr_field) or []:
        cutoff = _parse_date(smr.get(cfg.smr_date_key))
        targets = _collect_target_items(smr, cfg)
        contexts = _collect_context_before(records, cutoff, cfg)
        merged = _merge_entries(contexts, targets, cfg)
        if not merged:
            continue
        section = smr.get(cfg.smr_section_key) or example.get(cfg.record_section_key) or ""
        smr_type = smr.get(cfg.smr_type_key) or ""
        results.append(InstructionRecord(section=str(section), smr_type=str(smr_type), merged=merged))

    return results


def build_instruction_records(
    data_files: Sequence[str],
    split: str = "train",
    cache_dir: Optional[str] = None,
    cfg: Optional[PipelineConfig] = None,
) -> List[InstructionRecord]:
    cfg = cfg or PipelineConfig()
    dataset = load_dataset("json", data_files=list(data_files), split=split, cache_dir=cache_dir)

    records: List[InstructionRecord] = []
    for example in dataset:
        records.extend(_build_records_from_patient(example, cfg))
    return records


def prepare_instruction_dataset(
    data_files: Sequence[str],
    split: str = "train",
    cache_dir: Optional[str] = None,
    cfg: Optional[PipelineConfig] = None,
) -> Dataset:
    records = build_instruction_records(data_files, split=split, cache_dir=cache_dir, cfg=cfg)
    data = [record.to_dict() for record in records]
    if not data:
        return Dataset.from_list([])
    return Dataset.from_list(data)


def write_instruction_chunks(
    records: Sequence[InstructionRecord],
    output_dir: str,
    chunk_size: int = 500,
    prefix: str = "instruction_data_chunk",
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []
    for idx in range(0, len(records), chunk_size):
        chunk = [r.to_dict() for r in records[idx : idx + chunk_size]]
        if not chunk:
            continue
        path = os.path.join(output_dir, f"{prefix}_{idx // chunk_size + 1}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        paths.append(path)
        logger.info("Wrote %s records to %s", len(chunk), path)
    return paths


def _discover_files(data_dir: str) -> List[str]:
    files: List[str] = []
    for entry in sorted(os.listdir(data_dir)):
        if entry.lower().endswith(".json"):
            files.append(os.path.join(data_dir, entry))
    return files


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare instruction dataset from raw hospital notes")
    parser.add_argument("--data-files", nargs="*", default=None, help="JSON data files to load")
    parser.add_argument("--data-dir", default=None, help="Directory containing JSON files")
    parser.add_argument("--split", default="train", help="Dataset split when reading via datasets.load_dataset")
    parser.add_argument("--cache-dir", default=None, help="Optional cache directory for datasets")
    parser.add_argument("--output-dir", required=True, help="Directory to write chunked instruction JSON")
    parser.add_argument("--chunk-size", type=int, default=500, help="Number of records per chunk file")
    parser.add_argument("--prefix", default="instruction_data_chunk", help="Output filename prefix")
    return parser.parse_args(argv)


def _main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    files: List[str] = []
    if args.data_files:
        files.extend(args.data_files)
    if args.data_dir:
        files.extend(_discover_files(args.data_dir))

    if not files:
        raise SystemExit("No input files supplied via --data-files or --data-dir")

    records = build_instruction_records(files, split=args.split, cache_dir=args.cache_dir)
    if not records:
        logger.warning("No instruction records produced; nothing written")
        return

    write_instruction_chunks(records, args.output_dir, chunk_size=args.chunk_size, prefix=args.prefix)


if __name__ == "__main__":
    _main()
