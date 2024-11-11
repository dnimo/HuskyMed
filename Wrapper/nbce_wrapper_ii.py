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
        for line in contexts:
            if task_text == None:
                inputs.append("USER:" + line + "ASSISTANT:\n")
            else:
                inputs = ["USER:" + task_text + "ASSISTANT:\n"]
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
        input_ids = inputs.input_ids.to('cpu')
        attention_mask = inputs.attention_mask.to('cpu')
        processors = LogitsProcessorList([TopPLogitsWarper(0.95), TopKLogitsWarper(40)])
        preds = []
        past_key_values = None
        with torch.no_grad():
            chunk_size = 32
            for i in range(max_new_tokens):
                next_tokens_list = []
                for start_idx in range(0, n, chunk_size):
                    end_idx = min(start_idx + chunk_size, n)
                    input_ids_chunk = input_ids[start_idx:end_idx].to(self.device)
                    attention_mask_chunk = attention_mask[start_idx:end_idx].to(self.device)
                    outputs = self.model(
                        input_ids=input_ids_chunk,
                        attention_mask=attention_mask_chunk,
                        return_dict=True,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    if past_key_values is None:
                        past_key_values = outputs.past_key_values

                    # extract logits
                    beta = 0.75
                    logits = outputs.logits[:, -1]

                    # normalize logits
                    logits = logits - logits.logsumexp(dim=-1, keepdims=True)
                    logits = processors(input_ids, logits)

                    # calculate entropy
                    entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)

                    # find the token with the smallest entropy
                    k = entropy[1:].argmin() + 1
                    logits_max = logits[k]
                    logits_uncond = logits[0]

                    # merge logits
                    logits_merged = (1 + beta) * logits_max - beta * logits_uncond
                    logits = torch.where(logits_uncond > -100, logits_merged, logits_max)

                    # sample the next token
                    probas = torch.nn.functional.softmax(logits[None], dim=-1)
                    probas = torch.where(torch.isnan(probas), torch.zeros_like(probas), probas)
                    next_tokens_chunk = torch.multinomial(probas, num_samples=1).squeeze(1)
                    next_tokens_list.append(next_tokens_chunk)

                # concatenate the next tokens
                next_tokens = torch.cat(next_tokens_list, dim=0)

                # check if the next token is eos
                if next_tokens[0] == self.tokenizer.eos_token_id:
                    break

                ret = self.tokenizer.batch_decode(next_tokens)
                preds.append(ret[0])

                # prepare for the next iteration
                input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(n, 1, dtype=torch.long, device=attention_mask.device)], dim=-1)

            return "".join(preds)
