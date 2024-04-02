import torch
import torch.nn as nn


class BertEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, activation):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=0,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=True,
        )

    def forward(self, x):
        return self.transformer(x)
