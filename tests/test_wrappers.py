"""测试用的最小化wrapper与常量模拟, 供 experiment_manager 测试使用."""
from dataclasses import dataclass

from torch.nn import Module

# 常量模拟，与真实模块中的接口保持一致
PROMPTS = "prompts"
TARGET = "target"
TEXT_BETWEEN_SHOTS = "\n"


@dataclass
class PCWModelWrapper:
    """简化版PCW包装器用于测试."""
    model: Module
    tokenizer: object
    device: str
    context_window_size: int = 4096
    right_indentation: bool = False

    def pcw_generate(self, contexts, task_text, **kwargs):
        """模拟生成，返回固定文本用于测试."""
        return "测试生成输出"


@dataclass
class RestrictiveTokensLogitsProcessor:
    """简化版约束处理器用于测试."""
    restrictive_token_ids: object
    eos_token_id: int
    prompt_length_to_skip: int = 0