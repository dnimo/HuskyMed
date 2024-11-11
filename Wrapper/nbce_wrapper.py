from typing import Optional, List, Dict

import torch
from transformers import TopPLogitsWarper, LogitsProcessorList, TopKLogitsWarper, PreTrainedModel, \
    PreTrainedTokenizerBase


class NBCEModelWrapper:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 device: str,
                 window_size: int
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.window_size = window_size

    def get_contexts_cache(self, task_text, contexts):
        inputs = ["USER:" + task_text + "ASSISTANT:\n"]
        for line in contexts:
            inputs.append("USER:" + line + task_text + "ASSISTANT:\n")
        return inputs

    def nbce_generate(self,
                      contexts: Optional[List[str]] = None,
                      task_text: Optional[str] = None,
                      contexts_cache: Optional[Dict] = None,
                      max_new_tokens=800,
                      **kwargs
                      ) -> str:
        assert (contexts is None) != (
                contexts_cache is None), "nbce_generate should work with contexts or cache, not with both!"
        cache = contexts_cache or self.get_contexts_cache(task_text, contexts)
        inputs = self.tokenizer(cache, padding='max_length', truncation=True, max_length=self.window_size + 1,
                                return_tensors="pt")
        n = inputs['input_ids'].shape[0]
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        processors = LogitsProcessorList([TopPLogitsWarper(0.95), TopKLogitsWarper(40)])
        preds = []
        past_key_values = None
        for i in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            beta = 0.75
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
            if next_tokens[0] == self.tokenizer.eos_token_id:
                break

            ret = self.tokenizer.batch_decode(next_tokens)
            preds.append(ret[0])

            # prepare for next iteration
            input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(n, 1, dtype=torch.long, device=attention_mask.device)], dim=-1)

        return "".join(preds)
