from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Protocol


class TextTokenizer(Protocol):
    """Protocol for tokenizers used by metrics."""

    def tokenize(self, text: str) -> Sequence[str]: ...


@dataclass
class MecabTokenizer:
    """Thin wrapper around mecab-python3 with graceful fallback."""

    dictionary_path: str | None = None

    def __post_init__(self) -> None:
        try:
            import MeCab  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("mecab-python3 is required for MecabTokenizer") from exc
        args = ""
        if self.dictionary_path:
            args = f"-d {self.dictionary_path}"
        self._tagger = MeCab.Tagger(args)

    def tokenize(self, text: str) -> Sequence[str]:
        parsed = self._tagger.parse(text)
        if not parsed:
            return []
        tokens: list[str] = []
        for line in parsed.splitlines():
            if not line or line == "EOS":
                continue
            surface = line.split("\t", 1)[0]
            if surface:
                tokens.append(surface)
        return tokens


@dataclass
class WhitespaceTokenizer:
    """Fallback tokenizer when MeCab is unavailable."""

    def tokenize(self, text: str) -> Sequence[str]:
        return text.split()
