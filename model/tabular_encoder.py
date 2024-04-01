import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self):
        self.mlp = None  # TODO: fill this in

    def forward(self, x):
        return self.mlp(x)
