#! -*- coding: utf-8 -*-
import sys

sys.path.append("/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/huskyToolkit_0.2")

import torch
import json
import numpy as np
from utils import load_tokenizer, distribution_sampled, merge_short_sentences
from transformers import AutoModelForCausalLM
from transformers import TopPLogitsWarper, LogitsProcessorList, TopKLogitsWarper
from peft import PeftModel
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

Calm_Window_Size = 256

PROMPT_TEMPLATE = (
    "[INST] <<SYS>>\n"
    "あなたは{department}の医師です。 You are a medical doctor.\n"
    "<</SYS>>\n\n以下の臨床記録に基づいて、{smr_type}を作成してください。[/INST]"
)


def chunk_examples(items):
    samples = []
    refs = items['ref']
    for ref in refs:
        samples.append("記録カテゴリ：" + ref['data_type'] + ' ' + '記録内容：' + ref['text'] + '\n')
    return {'ref': samples, 'section': items['section'], 'smr_type': items['smr_type'], "smr": items['smr_text']}


def nbce_generate(nbce_model,
                  tokenizer,
                  context_texts: list[str],
                  task_texts: str,
                  max_new_tokens=800
                  ):
    inputs = []
    inputs.append(task_texts + "USER:" + "ASSISTANT:サマリー\n")
    for line in context_texts:
        inputs.append(task_texts + "USER:" + line + "ASSISTANT:サマリー\n")
    inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=Calm_Window_Size + 1,
                       return_tensors="pt")
    n = inputs['input_ids'].shape[0]
    input_ids = inputs.input_ids.to('cuda')
    attention_mask = inputs.attention_mask.to('cuda')
    processors = LogitsProcessorList([TopPLogitsWarper(0.95), TopKLogitsWarper(40)])
    preds = []
    past_key_values = None
    for i in range(max_new_tokens):
        outputs = nbce_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values

        beta = 0.65
        logits = outputs.logits[:, -1]
        logits = logits - logits.logsumexp(dim=-1, keepdims=True)
        logits = processors(input_ids, logits)
        entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
        k = entropy[1:].argmin() + 1
        logits_max = logits[k]
        logits_uncond = logits[0]
        logits_merged = (1 + beta) * logits_max - beta * logits_uncond
        logits = torch.where(logits_uncond > -100, logits_merged, logits_max)

        # 构建分布，采样
        probas = torch.nn.functional.softmax(logits[None], dim=-1)
        probas = torch.where(torch.isnan(probas), torch.zeros_like(probas), probas)
        next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)
        if next_tokens[0] == tokenizer.eos_token_id:
            break

        ret = tokenizer.batch_decode(next_tokens)
        preds.append(ret[0])

        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=attention_mask.device)],
                                   dim=-1)

    return "".join(preds)


if __name__ == "__main__":
    data_path = '/home/jovyan/public/zhang/train/test_data_230.json'
    model_path = '/home/jovyan/public/zhang/model/open-calm-7b'
    model_path2 = '/home/jovyan/public/zhang/model/calm2-7b-chat'
    peftmodel_path = '/home/jovyan/public/zhang/model/lora/R_8_S_1024_1e4_T_2024_7_25_instruction/'

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, peftmodel_path)
    tokenizer = load_tokenizer(model_path)

    data = load_dataset('json', data_files=data_path, split="train")
    chunk_data = data.map(chunk_examples)

    result = {}
    output_file_name = 'result_T_10_07'

    try:
        for rs in np.arange(0.15, 0.65, 0.05):
            for index, data in enumerate(chunk_data):
                task_text = PROMPT_TEMPLATE.format_map({'department': data['section'], 'smr_type': data['smr_type']})
                cache, length_distribution = distribution_sampled(data['ref'], tokenizer, sampling_rate=rs)
                cache = merge_short_sentences(cache, Calm_Window_Size)
                if len(cache) > 1:
                    nbce_out = nbce_generate(model, tokenizer, cache, task_text)
                    print(nbce_out, end='\n')
                    result[index] = {"ref": cache, "smr_human": data["smr"], "task": task_text, "nbce_out": nbce_out}

            with open(output_file_name + str(rs) + '.json', 'w', encoding='utf-8') as f:
                json.dump(result, f)
            print(f"Inference have been done & the result is stored at {output_file_name}")
    except Exception as e:
        # 打印错误信息
        print(f"An error occurred: {e}")

        with open(output_file_name + '_error.json', 'w', encoding='utf-8') as f:
            json.dump(result, f)
