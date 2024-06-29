import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel


class RoBERTa(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_hidden_layers=12,
        max_num_tokens=512,
    ):
        super().__init__()
        config = RobertaConfig(
            hidden_size=embed_dim,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_num_tokens,
            position_embedding_type="absolute",
        )
        self.model = RobertaModel(config, add_pooling_layer=False)

    def forward(self, x, mask):
        # mask applied in HF model class
        x = self.model(x, attention_mask=mask).last_hidden_state

        if self.mask_proportion:
            return x

        return x
