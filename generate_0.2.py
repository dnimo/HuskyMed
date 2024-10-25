#! -*- coding: utf-8 -*-
import sys

sys.path.append("/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/huskyToolkit_0.2")

import torch
import json
import numpy as np
from utils import load_tokenizer, distribution_sampled
from transformers import AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from utils import load_config
from datetime import datetime
import logging
from Wrapper import nbce_wrapper

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
device = "cuda" if torch.cuda.is_available() else "cpu"
config = load_config()

Calm_Window_Size = 64

logging.basicConfig(
    filename=config['log_path']['log_directory'],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def chunk_examples(items):
    samples = []
    refs = items['ref']
    for ref in refs:
        samples.append("記録カテゴリ：" + ref['data_type'] + ' ' + '記録内容：' + ref['text'] + '\n')
    samples = list(set(samples))
    return {'ref': samples, 'section': items['section'], 'smr_type': items['smr_type'], "smr": items['smr_text']}


if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained(config['model_path']['pretrained_model'], device_map='auto',
                                                 torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, config['model_path']['peft_model'])
    tokenizer = load_tokenizer(config['model_path']['pretrained_model'])

    data = load_dataset('json', data_files=config['data_paths']['test'], split="train")
    chunk_data = data.map(chunk_examples)

    nbce = nbce_wrapper.NBCEModelWrapper(model, tokenizer, device, Calm_Window_Size)

    result = {}
    output_path = config['data_paths']['output']

    try:

        for rs in np.arange(0.15, 0.65, 0.05):
            for index, data in enumerate(chunk_data):
                task_text = config['prompt_templates']['summarization'].format_map(
                    {'department': data['section'], 'smr_type': data['smr_type']})
                cache, length_distribution = distribution_sampled(data['ref'], tokenizer, sampling_rate=rs)
                # cache = merge_short_sentences(cache, Calm_Window_Size)
                if len(cache) > 1:
                    out = nbce.nbce_generate(contexts_cache=cache, task_text=task_text, max_new_tokens=800)
                    logging.info(f"Index: {index}, SMR: {data['smr']}, Task: {task_text}, NBCE: {out}")
                    result[index] = {"ref": cache, "smr_human": data["smr"], "task": task_text, "nbce_out": out}

            with open(output_path + f'result_{rs}_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(result, f)
            print(f"Inference have been done & the result is stored at {output_path}result_{rs}_{timestamp}.json")
    except Exception as e:
        # 打印错误信息
        logging.error(f"Error: {e}")

        with open(output_path + f'result_{timestamp}_error.json', 'w', encoding='utf-8') as f:
            json.dump(result, f)
            print(f"Error occurred & the result is stored at {output_path}result_{timestamp}_error.json")
