import os

import torch
import transformers
from datasets import load_dataset
from peft import get_peft_model, get_peft_model_state_dict, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_peft_available,
    logging,
)

from __init__ import GPTNeoX_Config

logger = logging.get_logger(__name__)
torch.cuda.empty_cache()

# config
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3
LEARNING_RATE = 8e-7
VAL_SET_SIZE = 20
window_size = 2048

OUTPUT_DIR = "/home/jovyan/public/zhang/model/lora/R_8_S_1024_1e4_T_2024_7_23_instruction/"

model = AutoModelForCausalLM.from_pretrained("/home/jovyan/public/zhang/model/open-calm-7b", device_map='balanced')
tokenizer = AutoTokenizer.from_pretrained("/home/jovyan/public/zhang/model/open-calm-7b")

model = get_peft_model(model, GPTNeoX_Config)

data_path = '/home/jovyan/public/zhang/train/instrucation/instrucation_data_chunk_85.json'
data = load_dataset('json', data_files=data_path, split='train')
data = data.train_test_split(test_size=100, shuffle=True, seed=42)

train_val = data['test'].train_test_split(
    test_size=VAL_SET_SIZE,
    shuffle=True,
    seed=42
)

train_data = train_val["train"]
val_data = train_val["test"]


class PeftTrainer(transformers.Trainer):

    def _load_from_peft_checkpoint(self, resume_from_checkpoint, model):
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        if not any(
                os.path.isfile(f) for f in [adapter_weights_file, adapter_safe_weights_file]
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")
        # Load adapters following PR # 24096 
        if is_peft_available() and isinstance(model, PeftModel):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint) or os.path.exists(resume_from_checkpoint):
                    model.load_adapter(resume_from_checkpoint, model.active_adapter)
                    # Load_adapter has no return value present, modify it when appropriate.
                    from torch.nn.modules.module import _IncompatibleKeys

                    load_result = _IncompatibleKeys([], [])
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"using `TrainerCallback` to save {ADAPTER_WEIGHTS_NAME} in corresponding folders, "
                        "here are some examples https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if model is None:
            model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if is_peft_available() and isinstance(model, PeftModel):
            # Try to load adapters before trying to load a torch model
            try:
                return self._load_from_peft_checkpoint(resume_from_checkpoint, model=model)
            except:
                return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            # If it is not a PeftModel, use the original _load_from_checkpoint
        else:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)


trainer = PeftTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        output_dir=OUTPUT_DIR,
        warmup_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_dir=OUTPUT_DIR + "/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        save_steps=100,
        eval_steps=100,
        save_total_limit=10,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

trainer.train(resume_from_checkpoint=False)
trainer.save_model(OUTPUT_DIR)

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

os.system("python /home/jovyan/zhang/huskyToolkit_0.2/generate.py")
