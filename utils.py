import json
import os
from collections import Counter

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, AutoTokenizer


# 读取 config.json 文件
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    input_dir = config["data_paths"]["input"]
    output_dir = config["data_paths"]["output"]
    log_dir = config["log_paths"]["log_directory"]
    for path in [input_dir, output_dir, log_dir]:
        os.makedirs(path, exist_ok=True)
    return config


def remove_duplicates(sentences):
    # 使用集合去重
    unique_sentences = list(set(sentences))

    # 保持原始顺序
    unique_sentences.sort(key=sentences.index)

    return unique_sentences


def merge_short_sentences(sentences: list, max_length=256) -> list:
    merged_sentences = []
    current_sentence = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if sentence_length > max_length:
            merged_sentences.append(sentence)
            continue
        if current_length + sentence_length < max_length:
            current_sentence.append(sentence)
            current_length += sentence_length
        else:
            merged_sentences.append("\n".join(current_sentence))
            current_sentence = [sentence]
            current_length = sentence_length

    # Add the last collected sentences
    if current_sentence:
        try:
            merged_sentences.append("\n".join(current_sentence))
        except Exception as e:
            print(current_sentence)

    return merged_sentences


def distribution_sampled(sentences: list, tokenizer: PreTrainedTokenizerBase, sampling_rate=0.65):
    sentences = remove_duplicates(sentences)
    # 获取每个句子的长度
    lengths = np.array([len(tokenizer(sentence)['input_ids']) for sentence in sentences])

    # 计长句短句出现次数
    length_counts = Counter(lengths)

    # 计算频率权重
    inverse_frequency_weights = np.array([1.0 / length_counts[length] for length in lengths])

    # 计算 softmax 分布
    length_tensor = torch.tensor(inverse_frequency_weights, dtype=torch.float32)
    length_distribution = torch.softmax(length_tensor, dim=0).numpy()

    length_distribution = np.clip(length_distribution, 1e-4, 1.0)
    length_distribution /= length_distribution.sum()

    # 计算采样大小
    sample_size = int(sampling_rate * len(sentences))

    # 检查非零项数是否足够进行采样
    non_zero_count = np.count_nonzero(length_distribution)
    if sample_size > non_zero_count:
        raise ValueError(
            f"Sample size {sample_size} is greater than the number of non-zero entries {non_zero_count} in the distribution.")

    # 根据分布进行采样
    sampled_indices = np.random.choice(
        len(sentences),
        size=sample_size,
        p=length_distribution,
        replace=False
    )

    # 获取采样的句子
    sampled_sentences = [sentences[i] for i in sampled_indices]

    return sampled_sentences, length_distribution


def select_and_merge_sentences(sentences, min_length_for_merge):
    # fix variable name typo and ensure dedup
    sentences = remove_duplicates(sentences)
    max_total_length = 40 * min_length_for_merge
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


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token
    tokenizer.padding_side = "left"
    return tokenizer
