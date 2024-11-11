import torch

# 启用自动混合精度和关闭梯度计算
with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    batch_size = input_ids.size(0)
    chunk_size = 32  # 可以调整这个值来优化显存使用
    max_new_tokens = 50

    for i in range(max_new_tokens):
        # 按照 chunk_size 分片处理
        next_tokens_list = []

        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)

            # 当前子批次
            input_ids_chunk = input_ids[start_idx:end_idx]
            attention_mask_chunk = attention_mask[start_idx:end_idx]

            # 前向传播
            outputs = self.model(
                input_ids=input_ids_chunk,
                attention_mask=attention_mask_chunk,
                return_dict=True,
                use_cache=True,
                past_key_values=past_key_values,
            )

            # 更新 past_key_values
            if past_key_values is None:
                past_key_values = outputs.past_key_values
            else:
                past_key_values = tuple(
                    torch.cat([pkv_old, pkv_new], dim=1) for pkv_old, pkv_new in
                    zip(past_key_values, outputs.past_key_values)
                )

            # 提取 logits
            beta = 0.75
            logits = outputs.logits[:, -1]

            # 正则化 logits
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            logits = processors(input_ids_chunk, logits)

            # 计算熵
            entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)

            # 找到熵最小的 token
            k = entropy[1:].argmin() + 1
            logits_max = logits[k]
            logits_uncond = logits[0]

            # 合并 logits
            logits_merged = (1 + beta) * logits_max - beta * logits_uncond
            logits = torch.where(logits_uncond > -100, logits_merged, logits_max)

            # 采样下一个 token
            probas = torch.nn.functional.softmax(logits[None], dim=-1)
            probas = torch.where(torch.isnan(probas), torch.zeros_like(probas), probas)
            next_tokens_chunk = torch.multinomial(probas, num_samples=1).squeeze(1)
            next_tokens_list.append(next_tokens_chunk)

        # 将所有子批次结果拼接
        next_tokens = torch.cat(next_tokens_list, dim=0)

        # 检查是否生成结束标记
        if next_tokens[0] == self.tokenizer.eos_token_id:
            break

        # 解码生成的 token
        ret = self.tokenizer.batch_decode(next_tokens)
        preds.append(ret[0])

        # 更新 input_ids 和 attention_mask，减少显存重复分配
        input_ids = next_tokens.unsqueeze(-1).expand(batch_size, 1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(batch_size, 1, dtype=torch.long, device=attention_mask.device)], dim=-1
        )
