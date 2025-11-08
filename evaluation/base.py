from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MetricResult:
    """Container for the scalar outputs produced by a metric."""

    name: str
    values: Dict[str, float]


class Metric(ABC):
    """Interface for text evaluation metrics."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def compute(self, reference: str, prediction: str) -> MetricResult:
        """Compute the metric for a single reference/prediction pair."""

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}(name={self.name!r})"
