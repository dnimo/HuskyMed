import os
import sys
from peft import LoraConfig

if sys.platform.startswith('win'):
    PATH_DATA = os.getcwd() + 'Unit\\'
if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
    PATH_DATA = os.getcwd() + 'Unit/'


# GPT-NeoX
GPTNeoX_Config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Llama
Llama_Config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)