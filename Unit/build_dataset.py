import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List

import datasets
import torch
import transformers
from datasets import load_dataset, concatenate_datasets

IGNORE_INDEX = -100

logger = logging.getLogger('__name__')

PROMPT_TEMPLATE = (
    "[INST] <<SYS>>\n"
    "あなたは{department}の医師です。 You are a medical doctor.\n"
    "<</SYS>>\n\n以下の臨床記録に基づいて、{smr_type}を作成してください。[/INST]"
)


def build_instruction_dataset(data_path: Union[List[str], str],
                              tokenizer: transformers.PreTrainedTokenizer,
                              max_seq_length: int, data_cache_dir=None,
                              preprocessing_num_workers=None,
                              ):
    def tokenization(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for section, smr_type, merged in zip(examples['section'], examples['smr_type'], examples['merged']):
            if merged is not None:
                for line in merged:
                    instruction = prompt.format_map({'department': section, 'smr_type': smr_type})
                    source = instruction + 'USER:記録カテゴリ：' + line[0] + ' ' + '記録内容：' + line[1]
                    target = 'ASSISTANT:サマリー\n' + line[2] + tokenizer.eos_token
                    sources.append(source)
                    targets.append(target)

        tokenized_sources = tokenizer(sources, return_attention_mask=False)
        tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            if len(s) > 1500:
                s = s[:1500]
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids': all_input_ids, 'labels': all_labels}
        return results

    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]
    for file in data_path:

        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
            cache_path = os.path.join(data_cache_dir, os.path.basename(file).split('.')[0] + f"_{max_seq_length}")
            os.makedirs(cache_path, exist_ok=True)

        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["section", "smr_type", "merged"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
