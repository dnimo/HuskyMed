import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import get_config

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


def build_instruction_dataset(
    data_path: Union[List[str], str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    data_cache_dir: str | None = None,
    preprocessing_num_workers: int | None = None,
):
    def tokenization(examples):
        sources: List[str] = []
        targets: List[str] = []
        cfg = get_config()
        prompt = cfg.get(
            "prompt_templates.summarization",
            "あなたは{department}の医師です。\n 以下の臨床記録に基づいて、{smr_type}を作成してください。",
        )
        for section, smr_type, merged in zip(examples["section"], examples["smr_type"], examples["merged"]):
            if merged is not None:
                for line in merged:
                    instruction = prompt.format_map({"department": section, "smr_type": smr_type})
                    source = instruction + "USER:記録カテゴリ：" + line[0] + " " + "記録内容：" + line[1]
                    target = "ASSISTANT:サマリー\n" + line[2] + tokenizer.eos_token
                    sources.append(source)
                    targets.append(target)

        tokenized_sources = tokenizer(sources, return_attention_mask=False)
        tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s, t in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            if len(s) > 1500:
                s = s[:1500]
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {"input_ids": all_input_ids, "labels": all_labels}
        return results

    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]
    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
            cache_path = os.path.join(data_cache_dir, os.path.basename(file).split(".")[0] + f"_{max_seq_length}")
            os.makedirs(cache_path, exist_ok=True)

        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info("training datasets-%s has been loaded from disk", file)
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["section", "smr_type", "merged"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format("torch")
        all_datasets.append(processed_dataset["train"])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def _extract_text(source: Dict[str, Any], primary_key: str, fallbacks: Sequence[str]) -> str:
    for key in (primary_key, *fallbacks):
        value = source.get(key)
        if value:
            return str(value)
    return ""


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
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
            return None
    return None


def _collect_patient_notes(
    row: pd.Series,
    *,
    pt_id_key: str,
    record_collection_key: str,
    record_items_key: str,
    record_timestamp_key: str,
    record_text_key: str,
    summary_collection_key: str,
    summary_items_key: str,
    summary_date_key: str,
    summary_text_key: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pt_id = row.get(pt_id_key)
    raw_records: List[Dict[str, Any]] = []
    for record in row.get(record_collection_key) or []:
        record_copy = dict(record)
        record_copy["_parsed_ts"] = _parse_timestamp(record.get(record_timestamp_key))
        raw_records.append(record_copy)
    raw_records.sort(key=lambda rec: rec.get("_parsed_ts") or datetime.max)

    summaries: List[Dict[str, Any]] = []
    for summary in row.get(summary_collection_key) or []:
        summary_copy = dict(summary)
        summary_copy["_parsed_ts"] = _parse_timestamp(summary.get(summary_date_key))
        summaries.append(summary_copy)
    summaries.sort(key=lambda smr: smr.get("_parsed_ts") or datetime.max)

    remaining_records = raw_records
    patient_records: List[Dict[str, Any]] = []
    patient_summaries: List[Dict[str, Any]] = []

    for summary in summaries:
        smr_date = summary.get("_parsed_ts")
        cutoff_index = len(remaining_records)
        if smr_date is not None:
            for idx, record in enumerate(remaining_records):
                rec_ts = record.get("_parsed_ts")
                if rec_ts is not None and rec_ts > smr_date:
                    cutoff_index = idx
                    break

        eligible_records = remaining_records[:cutoff_index]
        if not eligible_records:
            continue

        for record in eligible_records:
            meta = {
                "pt_id": pt_id,
                "timestamp": record.get("_parsed_ts"),
                "section": record.get("section"),
                "hcp_class": record.get("hcp_class"),
            }
            for note in record.get(record_items_key) or []:
                text = _extract_text(note, record_text_key, ("EMR_TEXT",))
                if not text:
                    continue
                entry = dict(meta)
                entry["text"] = text
                entry["data_type"] = note.get("data_type") or note.get("EMR_DATA_TYPE")
                patient_records.append(entry)

        base_summary_meta = {
            "pt_id": pt_id,
            "timestamp": smr_date,
            "section": summary.get("SECTION") or summary.get("section"),
            "smr_type": summary.get("EMR_TYPE") or summary.get("smr_type"),
        }
        for note in summary.get(summary_items_key) or []:
            text = _extract_text(note, summary_text_key, ("text",))
            if not text:
                continue
            entry = dict(base_summary_meta)
            entry["text"] = text
            entry["data_type"] = note.get("EMR_DATA_TYPE") or note.get("data_type")
            patient_summaries.append(entry)

        remaining_records = remaining_records[cutoff_index:]
        if not remaining_records:
            break

    return patient_records, patient_summaries


def build_tfidf_record_summary_pairs(
    data_path: Union[str, Sequence[str]],
    *,
    pt_id_key: str = "pt_id",
    record_collection_key: str = "records",
    record_items_key: str = "emr",
    record_timestamp_key: str = "ts",
    record_text_key: str = "text",
    summary_collection_key: str = "smrs",
    summary_items_key: str = "smr",
    summary_date_key: str = "INPUT_DATE",
    summary_text_key: str = "EMR_TEXT",
    min_similarity: float | None = None,
) -> List[Dict[str, Any]]:
    """Align clinical records with SMR sentences via TF-IDF cosine similarity."""

    files = [data_path] if isinstance(data_path, str) else list(data_path)
    if not files:
        return []

    dataset = load_dataset("json", data_files=files, split="train")
    frame = pd.DataFrame(dataset)

    matches: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        records, summaries = _collect_patient_notes(
            row,
            pt_id_key=pt_id_key,
            record_collection_key=record_collection_key,
            record_items_key=record_items_key,
            record_timestamp_key=record_timestamp_key,
            record_text_key=record_text_key,
            summary_collection_key=summary_collection_key,
            summary_items_key=summary_items_key,
            summary_date_key=summary_date_key,
            summary_text_key=summary_text_key,
        )
        record_texts = [entry["text"] for entry in records]
        summary_texts = [entry["text"] for entry in summaries]
        if not record_texts or not summary_texts:
            continue

        vectorizer = TfidfVectorizer()
        combined = record_texts + summary_texts
        tfidf = vectorizer.fit_transform(combined)
        records_matrix = tfidf[: len(record_texts)]
        summaries_matrix = tfidf[len(record_texts) :]
        similarity = cosine_similarity(records_matrix, summaries_matrix)
        if similarity.size == 0:
            continue

        best_indices = np.argmax(similarity, axis=1)
        for rec_idx, smr_idx in enumerate(best_indices):
            score = float(similarity[rec_idx, smr_idx])
            if min_similarity is not None and score < min_similarity:
                continue
            record_entry = records[rec_idx]
            summary_entry = summaries[smr_idx]
            matches.append(
                {
                    "pt_id": record_entry.get("pt_id"),
                    "record_text": record_entry["text"],
                    "record_type": record_entry.get("data_type"),
                    "record_section": record_entry.get("section"),
                    "record_timestamp": record_entry.get("timestamp"),
                    "smr_text": summary_entry["text"],
                    "smr_type": summary_entry.get("smr_type") or summary_entry.get("data_type"),
                    "smr_section": summary_entry.get("section"),
                    "smr_timestamp": summary_entry.get("timestamp"),
                    "similarity": score,
                }
            )

    return matches
