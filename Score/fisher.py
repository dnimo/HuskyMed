#! -*- coding: utf-8 -*-
import torch
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

Calm_Window_Size = 2048
model_path = '/home/jovyan/public/zhang/model/open-calm-7b/'
peftmodel_path = '/home/jovyan/public/zhang/model/lora/R_8_S_1000_2048_T_2024_3_1_graph_2.0'


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def get_log_probs(input_ids, model, past_key_values=None):
    outputs = model(input_ids, past_key_values=past_key_values)
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs, outputs.past_key_values


def compute_fisher_information(input_ids, model):
    log_probs, past_key_values = get_log_probs(input_ids, model)

    # 获取目标log_probs
    target_log_probs = log_probs.gather(2, input_ids_next.unsqueeze(-1)).squeeze(-1)

    # 计算梯度
    grads = []
    for i in range(target_log_probs.size(1)):
        model.zero_grad()
        target_log_probs[:, i].sum().backward(retain_graph=True)
        grads.append([param.grad.clone().detach() for param in model.parameters() if param.grad is not None])

    # 计算Fisher信息矩阵
    fisher_information = []
    for grad in grads:
        fisher_information.append([g ** 2 for g in grad])

    return fisher_information


def compute_confidence(fisher_information):
    # 计算每个token的置信度和不确定性
    confidences = []
    uncertainties = []
    for fisher_info in fisher_information:
        confidence = [torch.mean(f) for f in fisher_info]
        uncertainty = [torch.std(f) for f in fisher_info]
        confidences.append(confidence)
        uncertainties.append(uncertainty)
    return confidences, uncertainties


if __name__ == "__main__":

    context_texts = ["ZHANGは山に登るのがとても好きです", "ZHANGさんは京都大学に在学中です",
                     "ZHANGさんは中国山東省出身です"]
    task_texts = "張さんの紹介をお願いします"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    model = PeftModel.from_pretrained(model, peftmodel_path)
    tokenizer = load_tokenizer(model_path)

    inputs = []

    for line in context_texts:
        inputs.append(line + task_texts)
    inputs.append(task_texts)
    inputs = tokenizer(inputs, return_tensors="pt", padding='max_length', truncation=True, max_length=1024,
                       return_attention_mask=True)
    input_ids = inputs.input_ids.to(device)
    fisher_information = compute_fisher_information(input_ids, model)
    confidences, uncertainties = compute_confidence(fisher_information)
    with open("fisher_information.npy", "wb") as f:
        for i, token in enumerate(tokenizer.convert_ids_to_tokens(input_ids[0])):
            f.write(f"Token: {token}" + '\n')
            f.write(f"Confidence: {confidences[i]}" + '\n')
            f.write(f"Uncertainty: {uncertainties[i]}" + '\n')
