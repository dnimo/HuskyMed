#! -*- coding: utf-8 -*-
import sys

sys.path.append("/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/huskyToolkit_0.2")

import torch
import json
import re
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM
from transformers import TopPLogitsWarper, LogitsProcessorList
from processor.LogitsProcessor import RepetitionPenaltyLogitsProcessor
from peft import PeftModel
from datasets import load_dataset
from ParallelContextsWindows.pcw_wrapper import PCWModelWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

Calm_Window_Size = 256


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def chunk_examples(items):
    samples = []
    refs = items['ref']
    smr = "\n".join(items['smr_text'])
    for ref in refs:
        samples.append("記録カテゴリ：" + ref['data_type'] + ' ' + '記録内容：' + ref['text'] + '\n')
    return {'ref': samples, 'section': items['section'], 'smr_type': items['smr_type'], "smr": smr}


def pcw_generate(pcw_model, tokenizer, context_texts: list[str], task_texts: str, context_window_size,
                 right_indentation=False):
    pcw_model = PCWModelWrapper(pcw_model, tokenizer, device, context_window_size, right_indentation)
    result = pcw_model.pcw_generate(context_texts, task_texts)

    return result


def nbce_generate(nbce_model, tokenizer, context_texts: list[str], task_texts: str, max_new_tokens=800):
    processors = LogitsProcessorList([TopPLogitsWarper(0.75), RepetitionPenaltyLogitsProcessor()])
    inputs = []
    inputs.append(task_texts + "Response:")
    for line in context_texts:
        inputs.append(task_texts + line + "Response:")
    inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=Calm_Window_Size + 1,
                       return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    del inputs
    n = input_ids.shape[0]
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

        # ===== 核心代码开始 =====
        beta, eta = 0.75, 0.1
        logits = outputs.logits[:, -1]
        logits = logits - logits.logsumexp(dim=-1, keepdims=True)
        logits = processors(input_ids, logits)
        entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
        if i > 0:
            entropy[k] -= eta
        k = entropy[1:].argmin() + 1
        logits_max = logits[k]
        logits_uncond = logits[0]
        logits_merged = (1 + beta) * logits_max - beta * logits_uncond
        logits = torch.where(logits_uncond > -100, logits_merged, logits_max)
        # ===== 核心代码结束 =====

        # 构建分布，采样
        probas = torch.nn.functional.softmax(logits[None] / 1, dim=-1)
        probas = torch.where(torch.isnan(probas), torch.zeros_like(probas), probas)
        next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)
        if next_tokens[0] == tokenizer.eos_token_id:
            break

        ret = tokenizer.batch_decode(next_tokens)
        preds.append(ret[0])

        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=model.device)], dim=-1)

    return "".join(preds)


def select_and_merge_sentences(sentences, max_total_length=20 * Calm_Window_Size,
                               min_length_for_merge=Calm_Window_Size):
    selected_sentences = []
    total_length = 0

    left, right = 0, len(sentences) - 1
    current_sentence = ""

    while left <= right and total_length < max_total_length:
        if len(current_sentence) < min_length_for_merge:
            if total_length % 2 == 0:
                # 从左边取值
                current_sentence += sentences[left]
                left += 1
            else:
                # 从右边取值
                current_sentence += sentences[right]
                right -= 1
        else:
            # 当前句子长度已经超过或等于min_length_for_merge
            if total_length + len(current_sentence) <= max_total_length:
                selected_sentences.append(current_sentence)
                total_length += len(current_sentence)
                current_sentence = ""  # 重置当前句子
            else:
                break

    # 处理最后的剩余句子
    if current_sentence and total_length + len(current_sentence) <= max_total_length:
        selected_sentences.append(current_sentence)

    return selected_sentences


def filter_text(text):
    text = re.sub('## 要約.*\n', '', text)
    text = re.sub('\n\n.*\n', '', text)
    return text


def longlora_generate():
    pass


if __name__ == "__main__":
    data_path = '/home/jovyan/public/zhang/train/test_data_graph_3.0_100.json'
    model_path = '/home/jovyan/public/zhang/model/calm2-7b-chat'
    peftmodel_path = '/home/jovyan/public/zhang/model/lora/R_8_S_1024_1e4_T_2024_7_23_instruction/'

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='balanced')
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    model = PeftModel.from_pretrained(model, peftmodel_path)
    tokenizer = load_tokenizer(model_path)

    data = load_dataset('json', data_files=data_path, split="train")
    chunk_data = data.map(chunk_examples)

    result = {}
    try:
        for index, data in enumerate(chunk_data):
            task_text = 'あなたは' + data['section'] + 'の医師です。以下の臨床記録に基づいて' + data[
                'smr_type'] + 'を作成してください。' + '\n' + '患者の臨床記録：'
            cache = select_and_merge_sentences(data['ref'])
            if len(cache) > 1:
                nbce_out = nbce_generate(model, tokenizer, cache, task_text)
                nbce_out = filter_text(nbce_out)
                print('nbce_response:' + '\n')
                print(nbce_out, end='\n')

                # pcw method
                pcw_out = pcw_generate(model, tokenizer, cache, task_text, Calm_Window_Size, right_indentation=False)
                pcw_out = filter_text(pcw_out)
                print('pcw_response:' + '\n')
                print(pcw_out, end='\n')
                result[index] = {"ref": cache, "smr_human": data["smr"], "task": task_text, "nbce_out": nbce_out,
                                 "pcw_out": pcw_out}
        output_file = 'result_T_7_21.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f)
        print(f"Inference have been done & the result is stored at {output_file}")
    except Exception as e:
        # 打印错误信息
        print(f"An error occurred: {e}")
        output_file = 'result_T_7_21_error.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f)

        print(f"Inference have been done & the result is stored at {output_file}")
