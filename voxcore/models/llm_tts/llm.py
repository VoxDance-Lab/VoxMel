import logging

import torch
from transformers import Qwen2ForCausalLM, Qwen2Config
from voxcore.utils.nets_utils import make_pad_mask


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path, load_checkpoint=True):
        super().__init__()

        if load_checkpoint:
            self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)
        else:
            config = Qwen2Config.from_pretrained(pretrain_path)

            self.model = Qwen2ForCausalLM(config)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, maxlen=T).to(xs.device)
        outs = self.model.model(
            inputs_embeds=xs,
            attention_mask=masks,
            return_dict=True,
        )
        return outs.last_hidden_state, masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        outs = self.model.model(
            inputs_embeds=xs,
            attention_mask=masks,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.last_hidden_state
        new_cache = outs.past_key_values
        return xs, new_cache
