from __future__ import annotations

import warnings

# --- Optional flash-attn patches: safe import with fallback no-ops ---
try:
    from .attention import enable_gptneox_flash_attention, enable_llama_flash_attention
except Exception as e:
    # When flash_attn or CUDA is unavailable, expose no-op functions to avoid import-time crash.
    def enable_llama_flash_attention(*args, **kwargs) -> None:  # type: ignore
        warnings.warn(
            "FlashAttention for LLaMA not available (missing dependency or unsupported environment)."
            " Proceeding without enabling flash attention.",
            RuntimeWarning,
        )

    def enable_gptneox_flash_attention(*args, **kwargs) -> None:  # type: ignore
        warnings.warn(
            "FlashAttention for GPT-NeoX not available (missing dependency or unsupported environment)."
            " Proceeding without enabling flash attention.",
            RuntimeWarning,
        )

# --- Core generation utilities ---
from .generation import (
    NBCEGenerationConfig,
    NBCEModelWrapper,
    PCWContextCache,
    PCWModelWrapper,
    RestrictiveTokensLogitsProcessor,
    combine_past_key_values,
    generate_pcw_position_ids,
)

__all__ = [
    "enable_gptneox_flash_attention",
    "enable_llama_flash_attention",
    "NBCEGenerationConfig",
    "NBCEModelWrapper",
    "PCWContextCache",
    "PCWModelWrapper",
    "RestrictiveTokensLogitsProcessor",
    "combine_past_key_values",
    "generate_pcw_position_ids",
]
