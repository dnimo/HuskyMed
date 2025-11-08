from __future__ import annotations

from typing import Dict, Iterable, Sequence

from rouge_score import rouge_scorer

from .base import Metric, MetricResult
from .tokenizers import MecabTokenizer, TextTokenizer, WhitespaceTokenizer


class RougeMetric(Metric):
    """Wrapper around rouge_score.RougeScorer with MECAB support."""

    def __init__(
        self,
        variants: Sequence[str],
        *,
        tokenizer: str = "mecab",
        use_stemmer: bool = False,
    ) -> None:
        super().__init__(name="rouge")
        self._variants = list(dict.fromkeys(variants))
        if not self._variants:
            raise ValueError("RougeMetric requires at least one variant (e.g. rouge1)")
        self._tokenizer = self._build_tokenizer(tokenizer)
        self._scorer = rouge_scorer.RougeScorer(
            self._variants,
            use_stemmer=use_stemmer,
            tokenizer=self._tokenizer.tokenize if self._tokenizer else None,
        )

    @staticmethod
    def _build_tokenizer(name: str) -> TextTokenizer | None:
        if name == "mecab":
            try:
                return MecabTokenizer()
            except RuntimeError:
                return WhitespaceTokenizer()
        if name == "whitespace":
            return WhitespaceTokenizer()
        raise ValueError(f"Unknown tokenizer '{name}' for RougeMetric")

    def compute(self, reference: str, prediction: str) -> MetricResult:
        if not reference or not prediction:
            values = self._empty_scores()
            return MetricResult(metric=self.name, values=values)

        scores = self._scorer.score(reference, prediction)
        values: Dict[str, float] = {}
        for variant, triple in scores.items():
            values[f"{variant}_p"] = float(triple.precision)
            values[f"{variant}_r"] = float(triple.recall)
            values[f"{variant}_f"] = float(triple.fmeasure)
        return MetricResult(metric=self.name, values=values)

    def _empty_scores(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for variant in self._variants:
            values[f"{variant}_p"] = 0.0
            values[f"{variant}_r"] = 0.0
            values[f"{variant}_f"] = 0.0
        return values


def parse_rouge_variants(names: Iterable[str]) -> list[str]:
    return [name for name in names if name.startswith("rouge")]
