from peft import LoraConfig

from dataset_builder.datasets import (
    InstructionRecord,
    PipelineConfig,
    build_instruction_records,
    prepare_instruction_dataset,
    write_instruction_chunks,
)

__all__ = [
    "InstructionRecord",
    "PipelineConfig",
    "build_instruction_records",
    "prepare_instruction_dataset",
    "write_instruction_chunks",
    "GPTNeoX_Config",
    "Llama_Config",
]


# GPT-NeoX
GPTNeoX_Config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Llama
Llama_Config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)