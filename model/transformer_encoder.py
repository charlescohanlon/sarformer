import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()

        # Parameters taken from Google's BERT implementation
        # the nn.Transformer module includes sin/cos positional embeddings
        self.transformer = nn.Transformer(
            d_model=768,
            nhead=12,
            num_encoder_layers=12,
            num_decoder_layers=0,
            dim_feedforward=3072,
            dropout=0.5,
        )

    def forward(self, x):
        return self.transformer(x)
