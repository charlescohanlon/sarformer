import torch
import torch.nn as nn
from model.swinv2_encoder import Mlp


class Decoder(nn.Module):
    def __init__(self, num_heads=1, num_blocks=1, mask_proportion=0, mask_token=0):
        super().__init__()
        self.num_blocks = num_blocks
        # self.blocks = nn.ModuleList(

    def forward(self, x, enc_output):

        pass
