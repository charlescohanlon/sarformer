import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel


class RoBERTa(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_hidden_layers=12,
        max_num_tokens=512,
        mask_proportion=0.0,
        mask_token=0,
        ape=True,
    ):
        super().__init__()
        self.mask_proportion = mask_proportion
        self.mask_token = mask_token

        config = RobertaConfig(
            hidden_act=embed_dim,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_num_tokens,
            position_embedding_type="absolute" if ape else "relative_key",
        )
        self.model = RobertaModel(config, add_pooling_layer=False)

    def forward(self, x, mask):
        if self.mask_proportion:
            masked_full_x = self.mask(x, mask)

        # mask applied in HF model class
        x = self.model(x[0], attention_mask=mask).last_hidden_state

        if self.mask_proportion:
            return masked_full_x, x

        return x

    def mask(self, x, mask):
        masked_full_x = x.detach().clone()
        masked_full_x[~mask.to(torch.bool)] = self.mask_token

        return masked_full_x
