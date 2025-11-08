import math
from abc import ABC
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from transformers import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, apply_rotary_pos_emb, GPTNeoXLayer, \
    GPTNeoXModel, GPTNeoXForCausalLM

from .generation import generate_pcw_position_ids

"""
The following code is mainly copy+paste from the original modelling_llama.py:
LlamaAttention uses a caching mechanism for the positional rotation vectors (using LlamaRotaryEmbedding). 
This mechanism forces us to override LLaMa attention layer, which in turn forces us to override the decoder, 
and model (so that the correct forward function would be called).
"""


class GPTNeoXForCausalLMPCW(GPTNeoXForCausalLM, ABC):
    _no_split_modules = ["GPTNeoXLayerPCW"]
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config: GPTNeoXConfig):
        super(GPTNeoXForCausalLM, self).__init__(config)
        # using our model variant:
        self.gpt_neox = GPTNeoXModelPCW(config)

        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.LongTensor,
                                      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      windows_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      max_window_size: Optional[int] = None,
                                      sum_windows_size: Optional[int] = None,
                                      **kwargs
                                      ) -> Dict:
        """input_ids:
            ids of task_tokens.
         attention_mask:
            concatenation of windows + task tokens attentions masks.

         Note (past_key_values vs. windows_key_values):
             In the first token generation, past_key_values is None while windows_key_values contains the combined past
             key values of context windows. During the following generations, past_key_values is the concatenation of
             windows_key_values + previous generations. Thus, windows_key_values are practically ignored.
             """

        # only the last token for inputs_ids if past_key_values is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1:]
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create PCW's position_ids on the fly
            position_ids = generate_pcw_position_ids(attention_mask, max_window_size, past_key_values,
                                                     sum_windows_size, windows_key_values)

        if windows_key_values and not past_key_values:
            past_key_values = windows_key_values

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


class GPTNeoXModelPCW(GPTNeoXModel, ABC):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: GPTNeoXConfig):
        super(GPTNeoXModel, self).__init__(config)

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.emb_dropout = nn.Dropout(config.hidden_dropout)
        # using the alternative decoder layer:
        self.layers = nn.ModuleList([GPTNeoXLayerPCW(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class GPTNeoXLayerPCW(GPTNeoXLayer):
    def __init__(self, config: GPTNeoXConfig):
        super().__init__(config)
        # overriding attention:
        self.attention = GPTNeoXAttentionPCW(config=config)


class GPTNeoXAttentionPCW(GPTNeoXAttention):
    # we have to override the forward attention due to the rotary embeddings caching mechanism
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        # *** changes to the original code to accommodate PCW:
        # making sure that the model generates rotary embeddings in the correct length:
        seq_len = seq_len if position_ids is None else int(torch.max(position_ids) + 1)
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        # *** End of changes due to PCW, the rest of the function is copy-paste from the original transformer package.
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs