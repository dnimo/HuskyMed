import torch
import transformers
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import (
    logging,
)

from __init__ import GPTNeoX_Config
from Unit.build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset

logger = logging.get_logger(__name__)
torch.cuda.empty_cache()

# config
MICRO_BATCH_SIZE = 1
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3
LEARNING_RATE = 1e-4
VAL_SET_SIZE = 2000
window_size = 2048
OUTPUT_DIR = "/home/jovyan/public/zhang/model/lora/R_8_S_1024_1e4_T_2024_7_25_instruction/"
# model_path = "/home/jovyan/public/zhang/model/calm2-7b-chat"
model_path = "/home/jovyan/public/zhang/model/open-calm-7b"

# model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float32)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = get_peft_model(model, GPTNeoX_Config)

data_path = '/home/jovyan/public/zhang/train/instrucation/instrucation_data_chunk_85.json'
data = build_instruction_dataset(data_path, tokenizer=tokenizer, max_seq_length=window_size,
                                 preprocessing_num_workers=1)
data = data.train_test_split(test_size=80000, shuffle=True, seed=42)

train_val = data['test'].train_test_split(
    test_size=VAL_SET_SIZE,
    shuffle=True,
    seed=42
)

train_data = train_val["train"]
val_data = train_val["test"]

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        output_dir=OUTPUT_DIR,
        warmup_steps=20,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        logging_dir=OUTPUT_DIR + "/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        save_steps=100,
        eval_steps=100,
        save_total_limit=6
    ),
    data_collator=DataCollatorForSupervisedDataset(tokenizer),
)
model.config.use_cache = False

trainer.train()
trainer.save_model(OUTPUT_DIR)
