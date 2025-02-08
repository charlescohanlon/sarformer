from transformers import ConvNextV2Model, ConvNextV2Config
import torch
import torch.nn as nn
from einops import rearrange
from src.utils.timm.registry import register_model


class GatedMlp(nn.Module):
    """Implements SwiGLU and other gated feed-forward layers from Noam Shazeer's paper: https://arxiv.org/abs/2002.05202"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        # If gated, multiply hidden_dim by 2/3 to account for extra matmul
        hidden_features = int(2 * (hidden_features or in_features) / 3)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x):
        x = self.fc2(self.act(self.fc1(x)) * self.fc3(x))
        return x


class PointRegressionModel(nn.Module):
    def __init__(self, convnext_prefix: str, num_channels=4):
        super().__init__()
        self.num_channels = num_channels
        config = ConvNextV2Config.from_pretrained(convnext_prefix)
        config.num_channels = num_channels
        self.backbone = ConvNextV2Model(config)
        final_embed_dim = self.backbone.config.hidden_sizes[-1]
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AvgPool2d(kernel_size=(7, 7))
        self.head = GatedMlp(in_features=final_embed_dim, out_features=2)

    def forward(self, x, *vars):
        x = self.backbone(x).last_hidden_state
        x = self.gap(x)
        x = rearrange(x, "b c 1 1 -> b c")
        x = self.head(x)
        return x


class PointRegressionBaseline(nn.Module):
    def __init__(self, baseline_type: str, in_channels):
        if baseline_type != "center" and baseline_type != "uniform":
            raise ValueError("Unsupported baseline type:", baseline_type)
        super().__init__()
        self.baseline_type = baseline_type
        self._ = nn.Linear(in_channels, 1)

    def forward(self, x, *vars):
        B, _, H, W = x.shape
        y = torch.zeros((B, 2))
        if self.baseline_type == "center":
            y[:, 0] = H // 2
            y[:, 1] = W // 2
        elif self.baseline_type == "uniform":
            y[:, 0] = torch.randint(low=0, high=H, size=(B,))
            y[:, 1] = torch.randint(low=0, high=W, size=(B,))
        return y


@register_model
def convnext_gatedmlp_t(in_channels: int, **kwargs):
    return PointRegressionModel(
        convnext_prefix="facebook/convnextv2-tiny-22k-224",
        num_channels=in_channels,
    )


@register_model
def convnext_gatedmlp_b(in_channels: int, **kwargs):
    return PointRegressionModel(
        convnext_prefix="facebook/convnextv2-base-22k-224",
        num_channels=in_channels,
    )


@register_model
def convnext_gatedmlp_l(in_channels: int, **kwargs):
    return PointRegressionModel(
        convnext_prefix="facebook/convnextv2-large-22k-224",
        num_channels=in_channels,
    )


@register_model
def convnext_gatedmlp_h(in_channels: int, **kwargs):
    return PointRegressionModel(
        convnext_prefix="facebook/convnextv2-huge-1k-224",
        num_channels=in_channels,
    )


@register_model
def baseline_center(in_channels: int, **kwargs):
    return PointRegressionBaseline("center", in_channels)


@register_model
def baseline_uniform(in_channels: int, **kwargs):
    return PointRegressionBaseline("uniform", in_channels)
