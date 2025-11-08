from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import get_config
from model.base_loader import load_model_and_tokenizer
from model.wrappers import PCWModelWrapper, RestrictiveTokensLogitsProcessor
from model.wrappers.constants import PROMPTS, TARGET, TEXT_BETWEEN_SHOTS
from model.wrappers.utils import filter_extremely_long_samples, get_max_n_shots, plot_results_graph

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class ExperimentConfig:
    """User-controlled knobs for PCW few-shot experiments."""

    train_path: str
    test_path: str
    prompt_column: str = PROMPTS
    target_column: str = TARGET
    candidate_column: Optional[str] = None
    dataset_name: str = "dataset"
    output_dir: Optional[str] = None
    context_window_size: int = 4096
    right_indentation: bool = False
    shuffle_seed: int = 42
    use_restrictive_logits: bool = True
    stop_sequence: Optional[str] = None
    progress_bar: bool = True
    plot_results: bool = False
    max_test_samples: Optional[int] = None
    sample_with_replacement: bool = False
    model_path: Optional[str] = None
    peft_path: Optional[str] = None
    torch_dtype: Optional[str | torch.dtype] = "auto"
    generate_kwargs: Dict[str, object] = field(default_factory=dict)
    scorer: Optional[Callable[[str, str], Dict[str, float]]] = None


@dataclass
class ExperimentRunResult:
    """Structured output returned by :class:`ExperimentManager.run`."""

    per_example: pd.DataFrame
    per_n_shots: pd.DataFrame
    scores_matrix: np.ndarray
    n_shots: List[int]


class ExperimentManager:
    """High-level orchestration for PCW few-shot experiments.

    The manager loads datasets, prepares the PCW wrapper, performs few-shot
    prompting over the test set for a list of ``n_shots`` values, and aggregates
    simple metrics (default: exact match accuracy).
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.wrapper: Optional[PCWModelWrapper] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.max_n_shots: Optional[int] = None
        self.label_space: List[str] = []
        self._scorer = config.scorer or self._default_scorer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize_model(self) -> None:
        """Load model + tokenizer and wrap with :class:`PCWModelWrapper`."""

        cfg = get_config()
        base_model = self.config.model_path or cfg.get("model_paths.pretrained_model")
        peft_model = self.config.peft_path or cfg.get("model_paths.peft_model")
        model, tokenizer, device = load_model_and_tokenizer(
            model_path=base_model,
            peft_path=peft_model,
            torch_dtype=self.config.torch_dtype,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.wrapper = PCWModelWrapper(
            model=model,
            tokenizer=tokenizer,
            device=device,
            context_window_size=self.config.context_window_size,
            right_indentation=self.config.right_indentation,
        )
        _logger.info("PCW wrapper initialized (device=%s).", device)

    def load_datasets(self) -> None:
        """Load train/test tables, rename columns, and optionally trim size."""

        self.train_df = self._load_table(self.config.train_path)
        self.test_df = self._load_table(self.config.test_path)
        self._canonicalize_columns()
        if self.config.max_test_samples:
            self.test_df = self.test_df.head(self.config.max_test_samples).copy()
        if self.config.target_column:
            self.label_space = (
                self.train_df[self.config.target_column]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )

    def prepare(self) -> None:
        """Run tokenizer-aware filtering and derive ``max_n_shots``."""

        self._ensure_ready()
        assert self.tokenizer is not None  # for type checker
        self.train_df = filter_extremely_long_samples(self.train_df, self.tokenizer)
        self.test_df = filter_extremely_long_samples(self.test_df, self.tokenizer)
        self.max_n_shots = get_max_n_shots(
            self.train_df,
            self.test_df,
            self.tokenizer,
            prompt_size=self.config.context_window_size,
        )
        if self.max_n_shots <= 0:
            raise ValueError("No room for additional shots given the current window size.")
        _logger.info("Derived max_n_shots=%d", self.max_n_shots)

    def run(
        self,
        n_shots: Sequence[int],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> ExperimentRunResult:
        """Execute PCW decoding for all requested ``n_shots`` values."""

        self._ensure_ready()
        if not n_shots:
            raise ValueError("n_shots list must not be empty")
        if self.max_n_shots is None:
            raise RuntimeError("Call prepare() before run().")
        for k in n_shots:
            if k > self.max_n_shots:
                raise ValueError(f"Requested n_shots={k} exceeds computed max={self.max_n_shots}")

        records: List[Dict[str, object]] = []
        metric_names: Optional[List[str]] = None
        iterator = self.test_df.iterrows()
        progress: Iterable = tqdm(iterator, total=len(self.test_df), desc="test") if self.config.progress_bar else iterator

        for row_idx, row in progress:
            for shots in n_shots:
                contexts = self._build_contexts(shots, seed=row_idx)
                task_text = self._format_task_text(row)
                processor = self._build_restrictive_processor(row)
                kwargs = dict(self.config.generate_kwargs)
                kwargs.setdefault("temperature", temperature)
                if temperature == 0.0:
                    kwargs.setdefault("do_sample", False)
                else:
                    kwargs.setdefault("do_sample", True)
                output = self.wrapper.pcw_generate(
                    contexts=contexts,
                    task_text=task_text,
                    restrictive_processor=processor,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
                prediction = self._normalize_prediction(output)
                target = self._extract_target(row)
                metrics = self._scorer(prediction, target) if target is not None else {}
                metric_names = metric_names or list(metrics.keys())
                record = {
                    "index": row_idx,
                    "n_shots": shots,
                    PROMPTS: row[PROMPTS],
                    "prediction": prediction,
                    TARGET: target,
                }
                record.update(metrics)
                records.append(record)

        per_example = pd.DataFrame(records)
        if per_example.empty:
            raise RuntimeError("No predictions generated.")
        metric_names = metric_names or []
        aggregates = (
            per_example.groupby("n_shots")
            .agg({name: "mean" for name in metric_names} | {"index": "count"})
            .rename(columns={"index": "count"})
            .reset_index()
        )
        scores_matrix = self._build_scores_matrix(per_example, metric_names)
        if self.config.plot_results and metric_names:
            plot_results_graph(scores_matrix, self.config.dataset_name, list(n_shots))
        self._maybe_dump_predictions(per_example)
        return ExperimentRunResult(
            per_example=per_example,
            per_n_shots=aggregates,
            scores_matrix=scores_matrix,
            n_shots=list(n_shots),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_table(self, path: str) -> pd.DataFrame:
        suffix = Path(path).suffix.lower()
        if suffix in {".csv"}:
            return pd.read_csv(path)
        if suffix in {".jsonl", ".json"}:
            return pd.read_json(path, lines=suffix == ".jsonl")
        if suffix in {".parquet"}:
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file format: {path}")

    def _canonicalize_columns(self) -> None:
        assert self.train_df is not None and self.test_df is not None
        rename_map: Dict[str, str] = {}
        if self.config.prompt_column != PROMPTS:
            rename_map[self.config.prompt_column] = PROMPTS
        if self.config.target_column and self.config.target_column != TARGET:
            rename_map[self.config.target_column] = TARGET
        if rename_map:
            self.train_df.rename(columns=rename_map, inplace=True)
            self.test_df.rename(columns=rename_map, inplace=True)
        for column in [PROMPTS, TARGET]:
            if column not in self.train_df.columns:
                raise ValueError(f"Expected column '{column}' in training data")
            if column not in self.test_df.columns:
                raise ValueError(f"Expected column '{column}' in test data")

    def _ensure_ready(self) -> None:
        if self.wrapper is None or self.tokenizer is None or self.train_df is None or self.test_df is None:
            raise RuntimeError("Call initialize_model() and load_datasets() before running experiments.")

    def _build_contexts(self, n_shots: int, seed: int) -> List[str]:
        if n_shots == 0:
            return []
        assert self.train_df is not None
        rng = np.random.default_rng(self.config.shuffle_seed + seed)
        indices = self.train_df.index.to_numpy()
        if len(indices) == 0:
            raise ValueError("Training set is empty")
        if not self.config.sample_with_replacement and n_shots > len(indices):
            raise ValueError("Not enough training examples to sample without replacement")
        chosen = rng.choice(indices, size=n_shots, replace=self.config.sample_with_replacement)
        shots_df = self.train_df.loc[chosen]
        return [self._format_shot(row) for _, row in shots_df.iterrows()]

    def _format_shot(self, row: pd.Series) -> str:
        prompt = row[PROMPTS]
        target = str(row[TARGET])
        return f"USER:{prompt}{TEXT_BETWEEN_SHOTS}ASSISTANT:{target}\n"

    def _format_task_text(self, row: pd.Series) -> str:
        prompt = row[PROMPTS]
        return f"USER:{prompt}ASSISTANT:"

    def _extract_target(self, row: pd.Series) -> Optional[str]:
        if TARGET not in row or pd.isna(row[TARGET]):
            return None
        return str(row[TARGET])

    def _normalize_prediction(self, text: str) -> str:
        prediction = text.strip()
        if self.config.stop_sequence and self.config.stop_sequence in prediction:
            prediction = prediction.split(self.config.stop_sequence, 1)[0].strip()
        return prediction

    def _build_restrictive_processor(self, row: pd.Series) -> Optional[RestrictiveTokensLogitsProcessor]:
        if not self.config.use_restrictive_logits:
            return None
        assert self.tokenizer is not None and self.device is not None
        labels: Sequence[str]
        if self.config.candidate_column and self.config.candidate_column in row:
            labels_raw = row[self.config.candidate_column]
            labels = self._parse_labels(labels_raw)
        else:
            labels = self.label_space
        labels = [label for label in labels if label]
        if not labels:
            return None
        encoded = [self.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if not encoded:
            return None
        max_len = max(len(seq) for seq in encoded) + 1
        tensor = torch.full(
            (len(encoded), max_len),
            fill_value=self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        for idx, seq in enumerate(encoded):
            if seq:
                tensor[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
        return RestrictiveTokensLogitsProcessor(
            restrictive_token_ids=tensor,
            eos_token_id=self.tokenizer.eos_token_id,
            prompt_length_to_skip=0,
        )

    def _parse_labels(self, value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
            return [value]
        return [str(value)]

    def _default_scorer(self, prediction: str, target: Optional[str]) -> Dict[str, float]:
        if target is None:
            return {}
        return {"exact_match": float(prediction.strip() == target.strip())}

    def _build_scores_matrix(self, per_example: pd.DataFrame, metric_names: List[str]) -> np.ndarray:
        if not metric_names:
            return np.empty((0, 0))
        scores: List[np.ndarray] = []
        for shots, group in per_example.groupby("n_shots"):
            row = group[metric_names].to_numpy(dtype=float)
            scores.append(row.flatten())
        min_len = min(len(row) for row in scores)
        trimmed = [row[:min_len] for row in scores]
        return np.vstack(trimmed)

    def _maybe_dump_predictions(self, per_example: pd.DataFrame) -> None:
        output_dir = self.config.output_dir
        if not output_dir:
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        predictions_path = Path(output_dir) / f"{self.config.dataset_name}_predictions.csv"
        per_example.to_csv(predictions_path, index=False)
        _logger.info("Saved detailed predictions to %s", predictions_path)


__all__ = [
    "ExperimentConfig",
    "ExperimentManager",
    "ExperimentRunResult",
]
