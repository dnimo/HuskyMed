from __future__ import annotations

from sacrebleu.metrics import BLEU

from .base import Metric, MetricResult


class SacreBleuMetric(Metric):
    """Sentence-level SacreBLEU scorer returning scores in [0, 1]."""

    def __init__(self, *, smooth_method: str = "exp", smooth_value: float = 0.0) -> None:
        super().__init__(name="bleu")
        self._metric = BLEU(smooth_method=smooth_method, smooth_value=smooth_value)

    def compute(self, reference: str, prediction: str) -> MetricResult:
        if not reference or not prediction:
            return MetricResult(metric=self.name, values={"bleu": 0.0})
        score = self._metric.sentence_score(prediction, [reference])
        return MetricResult(metric=self.name, values={"bleu": score.score / 100.0})
