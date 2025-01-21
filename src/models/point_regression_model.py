from transformers import ConvNextConfig, ConvNextModel
import torch
import torch.nn as nn
from einops import rearrange
from src.utils.timm.registry import register_model


class PointRegressionModel(nn.Module):
    def __init__(self, num_channels=4, coordinate_dim=2):
        super().__init__()
        config = ConvNextConfig(num_channels=num_channels)
        self.model = ConvNextModel(config)
        for param in self.model.parameters():
            param.requires_grad = True
        self.num_channels = num_channels
        self.out_proj = nn.Conv2d(
            in_channels=768, out_channels=coordinate_dim, kernel_size=(7, 7)
        )

    def forward(self, x, *vars):
        x = self.model(x).last_hidden_state
        x = self.out_proj(x)
        x = rearrange(x, "b c 1 1 -> b c")  # remove spatial dims
        return x


@register_model
def convnext_t(
    in_channels: int,
    **kwargs,
):
    return PointRegressionModel(num_channels=in_channels)
