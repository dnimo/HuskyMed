from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .generation import NBCEModelWrapper, PCWModelWrapper


@dataclass
class GenerationResult:
    text: str
    strategy: str


class GenerationStrategy(Protocol):
    def generate(self, contexts: List[str], task_text: str, max_new_tokens: int = 512, **kwargs) -> str: ...


class NBCEStrategy:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: str, window_size: int):
        self.wrapper = NBCEModelWrapper(model, tokenizer, device, window_size)

    def generate(self, contexts: List[str], task_text: str, max_new_tokens: int = 512, **kwargs) -> str:
        return self.wrapper.nbce_generate(contexts=contexts, task_text=task_text, max_new_tokens=max_new_tokens, **kwargs)


class PCWStrategy:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: str, window_size: int, right_indentation: bool = False):
        self.wrapper = PCWModelWrapper(model, tokenizer, device, context_window_size=window_size, right_indentation=right_indentation)

    def generate(self, contexts: List[str], task_text: str, max_new_tokens: int = 512, **kwargs) -> str:
        return self.wrapper.pcw_generate(contexts=contexts, task_text=task_text, max_new_tokens=max_new_tokens, **kwargs)


class GenerationManager:
    def __init__(self, nbce: Optional[NBCEStrategy] = None, pcw: Optional[PCWStrategy] = None):
        self.nbce = nbce
        self.pcw = pcw

    def run(self, method: str, contexts: List[str], task_text: str, max_new_tokens: int = 512, **kwargs) -> GenerationResult:
        method = method.lower()
        if method == "nbce" and self.nbce:
            out = self.nbce.generate(contexts, task_text, max_new_tokens, **kwargs)
            return GenerationResult(text=out, strategy="nbce")
        elif method == "pcw" and self.pcw:
            out = self.pcw.generate(contexts, task_text, max_new_tokens, **kwargs)
            return GenerationResult(text=out, strategy="pcw")
        else:
            raise ValueError(f"Unknown or uninitialized method: {method}")
