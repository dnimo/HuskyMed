from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from transformers import (
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

__all__ = [
    "NBCEModelWrapper",
    "NBCEGenerationConfig",
    "PCWContextCache",
    "PCWModelWrapper",
    "RestrictiveTokensLogitsProcessor",
    "combine_past_key_values",
    "generate_pcw_position_ids",
]


@dataclass
class NBCEGenerationConfig:
    """Simple container that describes NBCE sampling preferences."""

    top_p: float = 0.95
    top_k: int = 40
    beta: float = 0.65
    temperature: float = 1.0


class NBCEModelWrapper:
    """Implements NBCE decoding as a thin adapter around a HF causal LM."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        window_size: int,
        generation_config: Optional[NBCEGenerationConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.window_size = window_size
        self.config = generation_config or NBCEGenerationConfig()

    @staticmethod
    def _build_prompts(task_text: str, contexts: Sequence[str]) -> List[str]:
        prompts = [f"USER:{task_text}ASSISTANT:\n"]
        for context in contexts:
            prompts.append(f"USER:{context}{task_text}ASSISTANT:\n")
        return prompts

    def build_prompt_cache(self, task_text: str, contexts: Sequence[str]) -> List[str]:
        """Return the textual prompts that will be tokenized for NBCE."""

        return self._build_prompts(task_text, contexts)

    def nbce_generate(
        self,
        contexts: Optional[Sequence[str]] = None,
        task_text: Optional[str] = None,
        contexts_cache: Optional[Sequence[str]] = None,
        max_new_tokens: int = 800,
        logits_processor: Optional[Iterable] = None,
    ) -> str:
        assert (contexts is None) != (
            contexts_cache is None
        ), "nbce_generate expects either raw contexts or a prebuilt cache."
        cache = list(contexts_cache or self.build_prompt_cache(task_text or "", contexts or []))

        encoded = self.tokenizer(
            cache,
            padding="max_length",
            truncation=True,
            max_length=self.window_size + 1,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        num_variants = input_ids.shape[0]

        processors = LogitsProcessorList(
            list(logits_processor) if logits_processor is not None else [
                TopPLogitsWarper(self.config.top_p),
                TopKLogitsWarper(self.config.top_k),
            ]
        )

        generated: List[str] = []
        past_key_values = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                logits = logits - logits.logsumexp(dim=-1, keepdim=True)
                logits = processors(input_ids, logits)

                entropy = -(logits.exp() * logits.clamp(min=-100, max=0)).sum(dim=-1)
                selector = entropy[1:].argmin() + 1 if logits.shape[0] > 1 else 0
                logits_max = logits[selector]
                logits_uncond = logits[0]
                merged = (1 + self.config.beta) * logits_max - self.config.beta * logits_uncond
                final_logits = torch.where(logits_uncond > -100, merged, logits_max)
                if self.config.temperature != 1.0:
                    final_logits = final_logits / self.config.temperature

                probabilities = torch.nn.functional.softmax(final_logits.unsqueeze(0), dim=-1)
                probabilities = torch.where(
                    torch.isnan(probabilities),
                    torch.zeros_like(probabilities),
                    probabilities,
                )
                next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)
                if next_tokens[0].item() == self.tokenizer.eos_token_id:
                    break

                fragments = self.tokenizer.batch_decode(next_tokens)
                generated.append(fragments[0])

                input_ids = next_tokens.unsqueeze(-1).repeat(num_variants, 1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(num_variants, 1, dtype=torch.long, device=self.device)],
                    dim=-1,
                )

        return "".join(generated)


@dataclass
class PCWContextCache:
    past_key_values: Tuple[Tuple[Tensor, Tensor], ...]
    past_attention_mask: Tensor
    max_window_size: int
    sum_windows_size: int


class RestrictiveTokensLogitsProcessor:
    """Restrict decoding to a predefined set of continuation token ids."""

    def __init__(
        self,
        restrictive_token_ids: torch.Tensor,
        eos_token_id: int,
        prompt_length_to_skip: int = 0,
        logits_bias: int = 100,
    ) -> None:
        if restrictive_token_ids.dim() != 2:
            raise ValueError("restrictive_token_ids must be (n, sequence_length)")
        self.restrictive_token_ids = restrictive_token_ids.clone()
        if not torch.all(self.restrictive_token_ids[:, -1] == eos_token_id):
            eos_column = torch.full(
                (self.restrictive_token_ids.size(0), 1),
                eos_token_id,
                dtype=self.restrictive_token_ids.dtype,
                device=self.restrictive_token_ids.device,
            )
            self.restrictive_token_ids = torch.cat([self.restrictive_token_ids, eos_column], dim=1)
        self.eos_token_id = eos_token_id
        self.logits_bias = logits_bias
        self.prompt_length_to_skip = prompt_length_to_skip
        self._mask = torch.ones(self.restrictive_token_ids.size(0), dtype=torch.bool)

    def update_new_prompt_length_to_skip(self, prompt_length_to_skip: int) -> None:
        self.prompt_length_to_skip = prompt_length_to_skip
        self._mask = torch.ones_like(self._mask, dtype=torch.bool)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(0) != 1:
            raise ValueError("RestrictiveTokensLogitsProcessor expects batch size 1")
        new_tokens = input_ids.size(1) - self.prompt_length_to_skip
        if new_tokens > 0:
            current = input_ids[0, -1].item()
            self._mask &= self.restrictive_token_ids[:, new_tokens - 1] == current
        scores[:, self.restrictive_token_ids[self._mask, new_tokens]] += self.logits_bias
        return scores


def combine_past_key_values(
    past_values: Sequence[Tuple[Tuple[Tensor, Tensor], ...]],
    longest_window_id: int,
) -> Tuple[Tuple[Tensor, Tensor], ...]:
    layers = len(past_values[0])
    anchor = past_values[longest_window_id]
    others = past_values[:longest_window_id] + past_values[longest_window_id + 1 :]
    combined: List[Tuple[Tensor, Tensor]] = []
    for layer in range(layers):
        keys = [anchor[layer][0]] + [entry[layer][0][:, :, 1:, :] for entry in others]
        values = [anchor[layer][1]] + [entry[layer][1][:, :, 1:, :] for entry in others]
        combined.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    return tuple(combined)  # type: ignore[return-value]


def generate_pcw_position_ids(
    attention_mask: Tensor,
    max_window_size: int,
    past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]],
    sum_windows_size: int,
    windows_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]],
) -> Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    num_task_tokens = position_ids.shape[1] - sum_windows_size
    position_ids[0, -num_task_tokens:] = torch.arange(
        max_window_size,
        max_window_size + num_task_tokens,
        device=position_ids.device,
    )
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past_key_values:
        return position_ids[:, -1].unsqueeze(-1)
    if windows_key_values:
        return position_ids[:, sum_windows_size:]
    return position_ids


class PCWModelWrapper:
    """Wrapper for PCW generation compatible with transformers.generate."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        context_window_size: int,
        right_indentation: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_window_size = context_window_size
        self.right_indentation = right_indentation

    @dataclass
    class _Window:
        text: str
        encoded_input: dict
        attention_mask: Tensor
        window_size: int
        past: Tuple[Tuple[Tensor, Tensor], ...]

    def _encode_windows(self, texts: Sequence[str]) -> List["PCWModelWrapper._Window"]:
        windows: List[PCWModelWrapper._Window] = []
        max_window_size = 0
        if self.right_indentation:
            max_window_size = max(
                int(self.tokenizer(text, add_special_tokens=True, return_length=True)["length"]) for text in texts
            )
        for text in texts:
            encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
            attention_mask = encoded["attention_mask"]
            if self.right_indentation:
                shift = max_window_size - attention_mask.shape[1]
                encoded["position_ids"] = attention_mask.cumsum(-1) - 1 + shift
            with torch.no_grad():
                output = self.model(**encoded)
            windows.append(
                PCWModelWrapper._Window(
                    text=text,
                    encoded_input=encoded,
                    attention_mask=attention_mask,
                    window_size=attention_mask.shape[1],
                    past=output.past_key_values,
                )
            )
        return windows

    def build_context_cache(self, contexts: Sequence[str]) -> PCWContextCache:
        if not contexts:
            raise ValueError("PCW generation requires at least one context segment.")
        windows = self._encode_windows(contexts)
        window_sizes = [window.window_size for window in windows]
        anchor_index = max(range(len(window_sizes)), key=window_sizes.__getitem__)
        combined_past = combine_past_key_values([window.past for window in windows], anchor_index)
        anchor_attention = windows[anchor_index].attention_mask
        attention_masks = [window.attention_mask[:, 1:] for idx, window in enumerate(windows) if idx != anchor_index]
        past_attention_mask = torch.cat([anchor_attention] + attention_masks, dim=1)
        total_window_tokens = sum(window_sizes) - (len(windows) - 1)
        return PCWContextCache(
            past_key_values=combined_past,
            past_attention_mask=past_attention_mask,
            max_window_size=max(window_sizes),
            sum_windows_size=total_window_tokens,
        )

    def pcw_generate(
        self,
        contexts: Optional[Sequence[str]] = None,
        task_text: Optional[str] = None,
        contexts_cache: Optional[PCWContextCache] = None,
        restrictive_processor: Optional[RestrictiveTokensLogitsProcessor] = None,
        **generate_kwargs,
    ) -> str:
        assert (contexts is None) != (
            contexts_cache is None
        ), "pcw_generate expects contexts or a cache, not both."
        cache = contexts_cache or self.build_context_cache(contexts or [])

        encoded_task = self.tokenizer(task_text or "", add_special_tokens=False, return_tensors="pt").to(self.device)
        if restrictive_processor is not None:
            restrictive_processor.update_new_prompt_length_to_skip(encoded_task["input_ids"].shape[1])
            current = generate_kwargs.get("logits_processor")
            if current is None:
                current = [restrictive_processor]
            elif isinstance(current, LogitsProcessorList):
                current = list(current) + [restrictive_processor]
            else:
                current = list(current) + [restrictive_processor]
            generate_kwargs["logits_processor"] = current

        attention_mask = torch.cat((cache.past_attention_mask, encoded_task["attention_mask"]), dim=1).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=encoded_task["input_ids"],
                attention_mask=attention_mask,
                windows_key_values=cache.past_key_values,
                max_window_size=cache.max_window_size,
                sum_windows_size=cache.sum_windows_size,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )[0]
        if generated[-1] == self.tokenizer.eos_token_id:
            generated = generated[:-1]
        prompt_length = encoded_task["input_ids"].shape[1]
        return self.tokenizer.decode(generated[prompt_length:])
