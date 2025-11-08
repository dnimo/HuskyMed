from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from config import get_config


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    # ensure tokens
    if tok.pad_token is None and hasattr(tok, "unk_token"):
        tok.pad_token = tok.unk_token
    tok.padding_side = "left"
    return tok


def load_base_model(model_name_or_path: str, torch_dtype: Optional[str | torch.dtype] = None) -> PreTrainedModel:
    dtype = torch_dtype or ("auto")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=dtype)
    return model


def apply_peft(model: PreTrainedModel, peft_path: Optional[str]) -> PreTrainedModel:
    if not peft_path:
        return model
    if not os.path.exists(peft_path):
        raise FileNotFoundError(f"PEFT path not found: {peft_path}")
    model = PeftModel.from_pretrained(model, peft_path)
    return model


def load_model_and_tokenizer(
    model_path: Optional[str] = None,
    peft_path: Optional[str] = None,
    torch_dtype: Optional[str | torch.dtype] = None,
):
    cfg = get_config()
    base_model_path = model_path or cfg.get("model_paths.pretrained_model")
    peft_model_path = peft_path or cfg.get("model_paths.peft_model")

    tok = load_tokenizer(base_model_path)
    model = load_base_model(base_model_path, torch_dtype=torch_dtype)
    model = apply_peft(model, peft_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, tok, device
