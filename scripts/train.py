#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified LoRA SFT training entrypoint for HuskyMed.

Reads defaults from config.json (hyperparameters, model paths) and allows CLI overrides.
Outputs a LoRA adapter (and trainer checkpoints) under --output-dir.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import transformers
import torch
from peft import get_peft_model

from config import get_config
from model.base_loader import load_tokenizer
from dataset_builder.datasets import DataCollatorForSupervisedDataset, build_instruction_dataset
from __init__ import GPTNeoX_Config, Llama_Config
from model.wrappers import enable_llama_flash_attention
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    hp = cfg.get("hyperparameters", {}) or {}
    default_model = cfg.get("model_paths.pretrained_model", "./models/pretrained/")

    p = argparse.ArgumentParser(description="LoRA instruction fine-tuning")
    p.add_argument("--base-model", type=str, default=default_model)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--data-path", type=str, required=True, nargs="+", help="one or more json files")
    p.add_argument("--micro-batch-size", type=int, default=int(hp.get("micro_batch_size", 1)))
    p.add_argument("--batch-size", type=int, default=int(hp.get("batch_size", 64)))
    p.add_argument("--epochs", type=int, default=int(hp.get("num_epochs", 3)))
    p.add_argument("--learning-rate", type=float, default=float(hp.get("learning_rate", 1e-4)))
    p.add_argument("--val-set-size", type=int, default=hp.get("val_set_size", 1000))
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--preprocessing-workers", type=int, default=1)
    p.add_argument("--lora-type", choices=["gptneox", "llama"], default="gptneox")
    p.add_argument("--task-type", choices=["instruction", "lm"], default="instruction",
                   help="instruction: supervised SFT with labels mask; lm: causal LM without supervision")
    p.add_argument("--text-field", type=str, default="text", help="field name for LM data JSON (when --task-type lm)")
    p.add_argument("--block-size", type=int, default=None, help="LM chunk size (defaults to --max-seq-len)")
    # LLaMA attention replacement switches (for training)
    p.add_argument("--llama-attn-replace", action="store_true", help="enable llama flash-attn replacement for training")
    p.add_argument("--llama-attn-full", action="store_true", help="use full flash attention (no grouped shift)")
    # Migrated tunable Trainer args from legacy scripts
    p.add_argument("--warmup-steps", type=int, default=int(hp.get("warmup_steps", 20)))
    p.add_argument("--logging-steps", type=int, default=int(hp.get("logging_steps", 100)))
    p.add_argument("--save-steps", type=int, default=int(hp.get("save_steps", 200)))
    p.add_argument("--eval-steps", type=int, default=int(hp.get("eval_steps", 200)))
    p.add_argument("--save-total-limit", type=int, default=int(hp.get("save_total_limit", 6)))
    p.add_argument("--resume-from-checkpoint", type=str, default=None, help="path to checkpoint directory to resume training")
    return p.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("./models/peft/", f"lora_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # tokenizer & dataset
    tokenizer = load_tokenizer(args.base_model)
    if args.task_type == "instruction":
        dataset = build_instruction_dataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_len,
            preprocessing_num_workers=args.preprocessing_workers,
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer)
    else:
        # Plain LM dataset: expect JSON with a text field; supports multiple files
        files = args.data_path
        ds = load_dataset('json', data_files=files, split='train')

        remove_cols = [c for c in ds.column_names if c != args.text_field]

        def tok_fn(batch):
            return tokenizer(batch[args.text_field], return_attention_mask=False)

        ds_tok = ds.map(tok_fn, batched=True, num_proc=args.preprocessing_workers, remove_columns=remove_cols)

        block_size = args.block_size or args.max_seq_len

        def group_texts(examples):
            # Concatenate
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        dataset = ds_tok.map(group_texts, batched=True, num_proc=max(1, args.preprocessing_workers))
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # train/val split
    if args.val_set_size and args.val_set_size > 0:
        data_split = dataset.train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
        train_data = data_split["train"]
        val_data = data_split["test"]
    else:
        train_data = dataset
        val_data = None

    # Optional LLaMA attention patch (training variant)
    if args.lora_type == "llama" and args.llama_attn_replace:
        enable_llama_flash_attention(mode="training", use_full=args.llama_attn_full)

    # model + LoRA
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
    )
    lora_cfg = GPTNeoX_Config if args.lora_type == "gptneox" else Llama_Config
    model = get_peft_model(model, lora_cfg)

    micro_bs = args.micro_batch_size
    grad_acc_steps = max(1, args.batch_size // max(1, micro_bs))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_bs,
            per_device_eval_batch_size=micro_bs,
            gradient_accumulation_steps=grad_acc_steps,
            output_dir=output_dir,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=args.logging_steps,
            evaluation_strategy="steps" if val_data is not None else "no",
            save_steps=args.save_steps,
            eval_steps=args.eval_steps if val_data is not None else None,
            save_total_limit=args.save_total_limit,
            fp16=False,
        ),
        data_collator=data_collator,
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)

    print(f"Training finished. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
