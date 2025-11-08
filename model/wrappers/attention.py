from __future__ import annotations

from typing import Literal

from .gptneox_attn_replace import replace_gpt_neox_attn as _raw_replace_gpt_neox
from .llama_attn_replace import replace_llama_attn as _raw_replace_llama_inference
from .llama_attn_replace_sft import replace_llama_attn as _raw_replace_llama_training

__all__ = [
    "enable_llama_flash_attention",
    "enable_gptneox_flash_attention",
]


def enable_llama_flash_attention(
    mode: Literal["training", "inference", "sft"] = "training",
    use_full: bool = False,
    inference_patch: bool = False,
) -> None:
    """Patch transformers' LLaMA attention with FlashAttention variants.

    Args:
        mode: "training" uses the standard patch, "sft" enables the fine-tuning specific patch,
            and "inference" falls back to the inference-safe variant.
        use_full: Controls whether to request the full attention path when available.
        inference_patch: Forwarded to the underlying patch to indicate inference paylod.
    """

    if mode in {"training", "sft"}:
        _raw_replace_llama_training(use_flash_attn=True, use_full=use_full, inference=inference_patch)
    elif mode == "inference":
        _raw_replace_llama_inference(use_flash_attn=True, use_full=use_full, inference=True)
    else:
        raise ValueError(f"Unsupported mode '{mode}'.")


def enable_gptneox_flash_attention(use_full: bool = False) -> None:
    """Patch GPT-NeoX attention with FlashAttention support."""

    _raw_replace_gpt_neox(use_flash_attn=True, use_full=use_full)
