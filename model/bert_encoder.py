import torch
import torch.nn as nn


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()

        # TODO: use word embeddings

        # Parameters taken from Google's BERT implementation at https://arxiv.org/pdf/1810.04805.pdf
        # (the nn.Transformer module includes sin/cos positional embeddings)
        self.transformer = nn.Transformer(
            d_model=768,
            nhead=12,
            num_encoder_layers=12,
            num_decoder_layers=0,
            dim_feedforward=3072,
            activation="gelu",
            batch_first=True,
        )

    def forward(self, x):
        return self.transformer(x)
