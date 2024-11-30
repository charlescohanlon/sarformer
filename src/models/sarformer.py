from typing import Tuple
import torch
from torch import nn
from src.utils.timm.registry import register_model
from src.models.unet import PatchedConvNeXtUNet
from transformers import T5EncoderModel


class SARFormer(nn.Module):

    def __init__(
        self,
        t5_model_name_or_path: str,
        in_channels: int = 4,
        out_channels: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        num_conv_blocks: int = 4,
        num_channels: int = 128,
        patch_size: int = 4,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 16),
        drop_path_rate: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        is_pretraining: bool = False,
    ):
        super().__init__()

        self.init_std = 0.02
        self.is_pretraining = is_pretraining

        self.seq_encoder = T5EncoderModel.from_pretrained(t5_model_name_or_path)
        if is_pretraining:
            for param in self.seq_encoder.parameters():
                param.requires_grad = False

        cond_dim = self.seq_encoder.config.hidden_size

        self.backbone = PatchedConvNeXtUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            cond_dim=cond_dim,
            num_conv_blocks=num_conv_blocks,
            model_channels=num_channels,
            channel_mult=channel_mult,
            drop_path_rate=drop_path_rate,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            act_layer=act_layer,
        )

    def forward(
        self,
        spatial_input: torch.Tensor,
        seq_input: torch.Tensor,
    ) -> torch.Tensor:
        context = self.seq_encoder(seq_input).last_hidden_state
        x = self.backbone(spatial_input, context)
        return x

    def train(self):
        self.seq_encoder.train()
        self.backbone.train()

    def eval(self):
        self.seq_encoder.eval()
        self.backbone.eval()

    def num_parameters_backbone(self):
        return self.backbone.num_parameters()

    def num_parameters_seq_encoder(self):
        return self.seq_encoder.num_parameters()

    def num_parameters(self):
        return self.num_parameters_backbone() + self.num_parameters_seq_encoder()


@register_model
def sarformer_b_mae(
    channels: int,
    **kwargs,
):
    model = SARFormer(
        t5_model_name_or_path="google/t5-v1_1-base",
        in_channels=channels,
        out_channels=channels,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_conv_blocks=8,
        num_channels=64,
        patch_size=4,
        channel_mult=(1, 2, 4, 8, 16),
        drop_path_rate=0.0,
        act_layer=nn.SiLU,
        is_pretraining=True,
        **kwargs,
    )
    return model
