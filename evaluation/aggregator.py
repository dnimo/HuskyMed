from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .base import Metric


@dataclass(slots=True)
class FieldAliases:
    """Preferred key order when retrieving reference or prediction text."""

    prediction: Tuple[str, ...] = ("prediction", "pred", "output")
    reference: Tuple[str, ...] = ("reference", "ref", "smr_human")


@dataclass(slots=True)
class EvaluationEngine:
    """Compute metrics for collections of prediction/reference pairs."""

    metrics: Sequence[Metric]
    field_aliases: FieldAliases = field(default_factory=FieldAliases)

    def score_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        reference, prediction = self._extract_texts(row)
        enriched: Dict[str, Any] = dict(row)
        for metric in self.metrics:
            result = metric.compute(reference, prediction)
            enriched.update(result.values)
        return enriched

    def score_rows(self, rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return [self.score_row(row) for row in rows]

    def aggregate(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        group_keys: Sequence[str],
        metric_keys: Sequence[str],
    ) -> List[Dict[str, Any]]:
        buckets: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
        for row in rows:
            key = tuple(row.get(k) for k in group_keys)
            buckets.setdefault(key, []).append(row)

        aggregated: List[Dict[str, Any]] = []
        for key, items in buckets.items():
            summary: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
            for metric_key in metric_keys:
                values = [self._to_float(it.get(metric_key)) for it in items]
                finite = [v for v in values if v is not None]
                if finite:
                    summary[metric_key] = sum(finite) / len(finite)
            summary["count"] = len(items)
            aggregated.append(summary)
        return aggregated

    def _extract_texts(self, row: Mapping[str, Any]) -> Tuple[str, str]:
        ref = self._first_match(row, self.field_aliases.reference)
        pred = self._first_match(row, self.field_aliases.prediction)
        return ref, pred

    @staticmethod
    def _first_match(row: Mapping[str, Any], fields: Sequence[str]) -> str:
        for field in fields:
            value = row.get(field)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number):  # pragma: no cover - defensive guard
            return None
        return number


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        header = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
