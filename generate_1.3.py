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
from Wrapper import nbce_wrapper_ii
from tqdm import tqdm

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
device = "cuda" if torch.cuda.is_available() else "cpu"
config = load_config()

Calm_Window_Size = 2048

logging.basicConfig(
    filename=config['log_path']['log_directory'],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def build_context(examples):
    ref_sample = []
    smr_sample = []
    for example in examples:
        prompt = config['prompt_templates']['summarization']
        sample = []
        smr_section = example['smrs']['SECTION']
        smr_type = example['smrs']['EMR_TYPE']
        smr_data_type_list = list(set([items['EMR_DATA_TYPE'] for items in example['smrs']['smr']]))
        smr = list(set([(smr['EMR_DATA_TYPE'], smr['EMR_TEXT']) for smr in example['smrs']['smr']]))
        for smr_data_type in smr_data_type_list:
            for record in example['records']:
                for line in record['emr']:
                    source = prompt.format_map({
                        "smr_section": smr_section,
                        "ref_section": record['section'],
                        "ref_hcp_class": record['hcp_class'],
                        "ref_type": line['type'],
                        "smr_type": smr_type,
                        "smr_data_type": smr_data_type,
                        "ref_text": line['text']
                    })
                    sample.append(source)
        sample = list(set(sample))
        ref_sample.extend(sample)
        smr_sample.append(smr)
    return {"ref": ref_sample, "smr": smr_sample}

if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained(config['model_path']['pretrained_model'], device_map='auto',
                                                 torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, config['model_path']['peft_model'])
    tokenizer = load_tokenizer(config['model_path']['pretrained_model'])

    data = load_dataset('json', data_files=config['data_paths']['test'], split="train")
    chunk_data = build_context(data)

    nbce = nbce_wrapper_ii.NBCEModelWrapper(model, tokenizer, device, Calm_Window_Size)

    result = {}
    output_path = config['data_paths']['output']


    try:
        for rs in np.arange(0.15, 0.65, 0.05):
            rs = round(rs, 2)
            filename = config['data_paths']['output'] + timestamp + '_' + str(rs) + '_.json'
            for index, data in tqdm(enumerate(zip(chunk_data['ref'], chunk_data['smr'])), desc=f"Generating as per sampling rate {rs}"):
                cache, length_distribution = distribution_sampled(data, tokenizer, sampling_rate=rs)
                if len(cache) > 1:
                    out = nbce.nbce_generate(contexts=cache, task_text=config['prompt_templates']['summarization'], max_new_tokens=800)
                    result[index] = {"ref": cache, "human":data[1], "nbce": out}
            with open(filename, 'w', 'w', encoding='utf-8') as f:
                    json.dump(result, f)
    except Exception as e:
        logging.error(f"Error: {e}")
        with open(output_path + f'result_{timestamp}_error.json', 'w', encoding='utf-8') as f:
            json.dump(result, f)
            print(f"Error occurred & the result is stored at {output_path}result_{timestamp}_error.json")
