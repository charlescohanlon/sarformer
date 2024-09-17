from abc import abstractmethod

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fourm.models.fm_utils import softmax1

from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a [B] Tensor of indices.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [B x N] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half).to(
        timesteps.device
    )

    args = timesteps.unsqueeze(1).float() * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                if isinstance(layer, CrossAttentionBlock):
                    x = layer(x, cond)
                else:
                    x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer using nearest-neighbor interpolation.

    :param scale_factor: the factor by which to upsample.
    :param resampling_mode: the mode of resampling.
    """

    def __init__(self, scale_factor=2, resampling_mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = resampling_mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

    def extra_repr(self):
        return f"scale_factor={self.scale_factor}, mode={self.mode}"


class Downsample(nn.Module):
    """
    A downsampling layer using average pooling.

    :param scale_factor: the factor by which to downsample.
    """

    def __init__(self, scale_factor=1 / 2):
        super().__init__()
        self.scale_factor = scale_factor
        reciprocal = int(1 / scale_factor)
        self.pool = nn.AvgPool2d(kernel_size=reciprocal, stride=reciprocal)

    def forward(self, x):
        return self.pool(x)

    def extra_repr(self):
        return f"scale_factor={self.scale_factor}"


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Implementation from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob

    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvNeXtBlock(TimestepBlock):
    """
    ConvNeXt Block. There are two equivalent implementations:
    Modified from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Args:
        dim (int): Number of input channels.
        time_embed_dim (int): Size of timestep embedding dimension.
        output_dim (int): Number of output channels. Default: None (which results in same as dim).
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (int): Ratio of effective mlp hidden dim to embedding dim. Default: 4.
        act_layer (nn.Module): Activation layer. Default: nn.GELU.
    """

    def __init__(
        self,
        dim,
        time_embed_dim,
        output_dim=None,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = dim
        self.time_embed_proj = nn.Linear(time_embed_dim, dim)
        self.skip_proj = (
            nn.Linear(dim, output_dim) if dim != output_dim else nn.Identity()
        )

        # depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise/1x1 convs, implemented with linear layers
        hidden_dim = int(mlp_ratio * dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.pwconv2 = nn.Linear(hidden_dim, output_dim)

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((output_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, emb):
        emb = self.time_embed_proj(emb)
        h = self.skip_proj(rearrange(x, "b c h w -> b h w c"))

        x = x + emb[..., None, None]  # (B, C, H, W) + (B, C, 1, 1) -> (B, C, H, W)
        x = self.dwconv(x)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, "b h w c -> b c h w")

        x = self.drop_path(x) + rearrange(h, "b h w c -> b c h w")
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = cond_dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(cond_dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        _, N, _ = x.shape
        _, M, _ = cond.shape

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(cond)
            .reshape(B, M, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = softmax1(attn, dim=-1)  # NOTE: we're using the custom softmax here
        attn = self.attn_drop(attn)

        h = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        h = self.proj(h)
        h = self.proj_drop(h)
        return (x + h).reshape(B, C, H, W)


class ConvNeXtUNetModel(ModelMixin, ConfigMixin):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param cond_dim: the dimension of the conditioning tensor.
    :param num_conv_blocks: number of residual blocks per downsample. Must be at least 2.
    :param attn_drop: the dropout probability in the attention mechanism.
    :param proj_drop: the dropout probability in the projection after the attention.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param num_heads: the number of attention heads in each attention layer.
    :param qkv_bias: include bias in the attention QKV projection.
    """

    def __init__(
        self,
        in_channels=3,
        model_channels=256,
        out_channels=3,
        cond_dim=768,
        num_conv_blocks=3,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0,
        channel_mult=(1, 2, 4, 8, 16),
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.model_channels = model_channels
        time_embed_dim = int(model_channels * mlp_ratio)
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            act_layer(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.in_proj = nn.Linear(in_channels, model_channels)

        total_num_conv_blocks = 2 * (len(channel_mult) - 1) * num_conv_blocks
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_num_conv_blocks)
        ]
        dp_index = 0

        # Downsampling
        self.down_blocks = nn.ModuleList([])
        down_chans = []

        # for (1, 2, 4, 8, 16) this zip() produces (1, 2), (2, 4), (4, 8)
        # so the last mult is the middle block
        for level, next_level in zip(channel_mult[:-2], channel_mult[1:-1]):
            # decrease spacial dims
            layers = [Downsample(scale_factor=level / next_level)] if level > 1 else []
            level_chans = model_channels * level
            next_level_chans = model_channels * next_level
            layers.extend(
                [
                    *[
                        # isotropic
                        ConvNeXtBlock(
                            dim=level_chans,
                            time_embed_dim=time_embed_dim,
                            drop_path=dp_rates[dp_index + i],
                            mlp_ratio=mlp_ratio,
                            act_layer=act_layer,
                        )
                        for i in range(num_conv_blocks - 1)
                    ],
                    # increase embedding dims
                    ConvNeXtBlock(
                        dim=level_chans,
                        time_embed_dim=time_embed_dim,
                        output_dim=next_level_chans,
                        drop_path=dp_rates[dp_index + num_conv_blocks - 1],
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                    ),
                    CrossAttentionBlock(
                        dim=next_level_chans,
                        cond_dim=cond_dim,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        qkv_bias=qkv_bias,
                    ),
                ]
            )
            dp_index += num_conv_blocks
            self.down_blocks.append(TimestepEmbedSequential(*layers))
            down_chans.append(next_level_chans)

        # Middle
        last_level_chans = model_channels * channel_mult[-1]
        self.middle_block = TimestepEmbedSequential(
            Downsample(scale_factor=channel_mult[-2] / channel_mult[-1]),
            *[
                # isotropic
                ConvNeXtBlock(
                    dim=next_level_chans,
                    time_embed_dim=time_embed_dim,
                    drop_path=dp_rates[dp_index + i],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                )
                for i in range(num_conv_blocks - 1)
            ],
            # increase embedding dims
            ConvNeXtBlock(
                dim=next_level_chans,
                time_embed_dim=time_embed_dim,
                output_dim=last_level_chans,
                drop_path=dp_rates[dp_index + num_conv_blocks - 1],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
            ),
            CrossAttentionBlock(
                dim=last_level_chans,
                cond_dim=cond_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                qkv_bias=qkv_bias,
            ),
            *[
                # isotropic
                ConvNeXtBlock(
                    dim=last_level_chans,
                    time_embed_dim=time_embed_dim,
                    drop_path=dp_rates[dp_index + num_conv_blocks + i],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                )
                for i in range(num_conv_blocks - 1)
            ],
            # decrease embedding dims
            ConvNeXtBlock(
                dim=last_level_chans,
                time_embed_dim=time_embed_dim,
                output_dim=next_level_chans,
                drop_path=dp_rates[dp_index + num_conv_blocks * 2 - 1],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
            ),
            Upsample(scale_factor=channel_mult[-1] / channel_mult[-2]),
        )
        dp_index += 2 * num_conv_blocks

        # Upsampling
        self.up_blocks = nn.ModuleList([])

        # for (1, 2, 4, 8, 16) this produces (8, 4), (4, 2), (2, 1)
        for prev_level, level in list(zip(channel_mult[1:-1], channel_mult[:-2]))[::-1]:
            prev_level_chans = model_channels * prev_level
            level_chans = model_channels * level
            layers = [
                ConvNeXtBlock(
                    dim=prev_level_chans + down_chans.pop(),
                    time_embed_dim=time_embed_dim,
                    # if only one conv block (so this is the last) reduce chans
                    output_dim=(
                        level_chans if num_conv_blocks == 1 else prev_level_chans
                    ),
                    drop_path=dp_rates[dp_index],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                ),
                *[
                    ConvNeXtBlock(
                        dim=prev_level_chans,
                        time_embed_dim=time_embed_dim,
                        # reduces chans if the last conv block, otherwise isotropic
                        output_dim=level_chans if i == num_conv_blocks - 2 else None,
                        drop_path=dp_rates[dp_index + 1 + i],
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                    )
                    for i in range(num_conv_blocks - 1)
                ],
                CrossAttentionBlock(
                    dim=level_chans,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    qkv_bias=qkv_bias,
                ),
            ]
            dp_index += num_conv_blocks
            if level > 1:  # don't upsample again at the top level
                layers.append(Upsample(scale_factor=prev_level / level))
            self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.norm = nn.LayerNorm(level_chans, eps=1e-6)
        self.out_proj = nn.Linear(level_chans, out_channels)

    def forward(self, x, timesteps, cond):
        """
        Apply the model to an input batch.

        :param x: an [B x C x H x W] Tensor of inputs.
        :param timesteps: a [B] Tensor of timesteps or a number of them.
        :param cond: an [B x N x D] Tensor of encoded conditioning.
        :return: an [B x C x H x W] Tensor of outputs.
        """
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps.unsqueeze(0).to(x.device)

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        hs = []  # Hidden states for skip connections

        x = self.in_proj(rearrange(x, "b c h w -> b h w c"))
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")

        for blk in self.down_blocks:
            x = blk(x, emb, cond)
            hs.append(x)

        x = self.middle_block(x, emb, cond)

        for blk in self.up_blocks:
            x = torch.cat([x, hs.pop()], dim=1)
            x = blk(x, emb, cond)

        x = self.norm(rearrange(x, "b c h w -> b h w c"))
        x = self.out_proj(x)
        return rearrange(x, "b h w c -> b c h w")

    def __len__(self):
        return sum(
            [
                1
                for m in self.modules()
                if isinstance(m, ConvNeXtBlock) or isinstance(m, CrossAttentionBlock)
            ]
        )


class PatchedConvNeXtUNet(ConvNeXtUNetModel):
    """Patched UNet with conditioning upsampled and applied via cross-attention.
    For more details, see https://arxiv.org/abs/2207.04316

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        cond_channels: Number of conditioning channels
        patch_size: Size of the patch projection before and after the UNet
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        *args,
        **kwargs,
    ):
        in_channels_p = in_channels * patch_size**2
        out_channels_p = out_channels * patch_size**2
        super().__init__(
            in_channels=in_channels_p,
            out_channels=out_channels_p,
            *args,
            **kwargs,
        )
        self.P_H, self.P_W = patch_size, patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        sample: torch.FloatTensor,  # Shape (B, C, H, W)
        timesteps: Union[torch.Tensor, float, int],  # Shape (B) if tensor
        cond: torch.Tensor = None,  # Shape (B, N, D)
    ):
        _, _, H, W = sample.shape

        assert (
            H % self.P_H == 0 and W % self.P_W == 0
        ), f"Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}"

        N_H, N_W = H // self.P_H, W // self.P_W  # Number of patches in height and width

        # Patchify input from B C H W -> B (C * P_H * P_W) N_H N_W
        x = rearrange(
            sample,
            "b c (nh ph) (nw pw) -> b (c ph pw) nh nw",
            ph=self.P_H,
            pw=self.P_W,
            nh=N_H,
            nw=N_W,
        )

        x = super().forward(x, timesteps, cond=cond)

        # Depatchify output from B (C * P_H * P_W) N_H N_W -> B C H W
        x = rearrange(
            x,
            "b (c ph pw) nh nw -> b c (nh ph) (nw pw)",
            ph=self.P_H,
            pw=self.P_W,
            nh=N_H,
            nw=N_W,
        )

        return x

    def __len__(self):
        return super().__len__()