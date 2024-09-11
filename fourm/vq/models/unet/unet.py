# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractmethod
from typing import Union, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fourm.models.fm_utils import softmax1

from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

from .nn import (
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
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
                # to support xattn in most of the sequential blocks used in unet
                # no attention blocks are also timestep ones
                if isinstance(layer, CrossAttentionBlock):
                    x = layer(x, cond)
                else:
                    x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, in_channels, use_conv, signal_dim=2, out_channels=None):
        super().__init__()
        self.channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.dims = signal_dim
        if use_conv:
            self.conv = conv_nd(
                signal_dim, self.channels, self.out_channels, 3, padding=1
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param in_channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param signal_dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, in_channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.in_channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.in_channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param in_channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param signal_dim: determines if the signal is 1D, 2D, or 3D.
    :param upsample: if True, use this block for upsampling.
    :param downsample: if True, use this block for downsampling.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        signal_dim=2,
        upsample=False,
        downsample=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(signal_dim, in_channels, self.out_channels, 3, padding=1),
        )

        self.updown = upsample or downsample

        if upsample:
            self.h_upd = Upsample(in_channels, False, signal_dim)
            self.x_upd = Upsample(in_channels, False, signal_dim)
        elif downsample:
            self.h_upd = Downsample(in_channels, False, signal_dim)
            self.x_upd = Downsample(in_channels, False, signal_dim)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(signal_dim, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                signal_dim, in_channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(
                signal_dim, in_channels, self.out_channels, 1
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        _, N, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = softmax1(attn, dim=-1)
        attn = self.attn_drop(attn)

        h = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        h = self.proj(h)
        h = self.proj_drop(h)
        return (x + h).reshape(B, C, H, W)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

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


class UNetModel(ModelMixin, ConfigMixin):
    """
    The full UNet model with attention and timestep embedding.

    :param image_size: the size of the input image.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param attn_drop: the dropout probability in the attention mechanism.
    :param proj_drop: the dropout probability in the projection after the attention.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param signal_dim: determines if the signal is 1D, 2D, or 3D.
    :param cond_type: the method used to apply conditioning (either 'cat' or 'xattn').
    :param num_heads: the number of attention heads in each attention layer.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        image_size=224,
        in_channels=3,
        model_channels=256,
        out_channels=3,
        num_res_blocks=3,
        attention_resolutions=[8, 16],
        attn_drop=0.0,
        proj_drop=0.0,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        signal_dim=2,
        cond_type="cat",
        num_heads=1,
        use_scale_shift_norm=False,
        resblock_updown=False,
    ):
        super().__init__()
        if cond_type not in ("cat", "xattn"):
            raise ValueError(f"Unknown cond_type {cond_type}")

        self.image_size = image_size
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.cond_type = cond_type
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Downsampling
        self.input_blocks = nn.ModuleList([])
        input_block_chans = []
        for level, mult in enumerate(channel_mult):
            downsample_rate = mult
            level_chans = model_channels * mult

            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        in_channels=level_chans,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=level_chans,
                        signal_dim=signal_dim,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                if downsample_rate in attention_resolutions:
                    layers.append(
                        SelfAttentionBlock(
                            dim=level_chans,
                            num_heads=num_heads,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                        )
                        if cond_type == "cat"
                        else CrossAttentionBlock(
                            dim=level_chans,
                            num_heads=num_heads,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += level_chans
                input_block_chans.append(level_chans)

            if level < len(channel_mult) - 1:
                downsampled_chans = model_channels * channel_mult[level + 1]
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            in_channels=level_chans,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=downsampled_chans,
                            signal_dim=signal_dim,
                            use_scale_shift_norm=use_scale_shift_norm,
                            downsample=True,
                        )
                        if resblock_updown
                        else Downsample(
                            in_channels=level_chans,
                            use_conv=conv_resample,
                            dims=signal_dim,
                            out_channels=downsampled_chans,
                        )
                    )
                )
                input_block_chans.append(downsampled_chans)
                self._feature_size += downsampled_chans

        # Middle
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                in_channels=level_chans,
                emb_channels=time_embed_dim,
                dropout=dropout,
                signal_dim=signal_dim,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                SelfAttentionBlock(
                    dim=level_chans,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                if self.cond_type == "cat"
                else CrossAttentionBlock(
                    dim=level_chans,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
            ),
            ResBlock(
                in_channels=level_chans,
                emb_channels=time_embed_dim,
                dropout=dropout,
                signal_dim=signal_dim,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            downsample_rate = mult
            level_chans = model_channels * mult

            for _ in range(num_res_blocks):
                skip_connection_chans = input_block_chans[level]
                level_and_skip_chans = level_chans + skip_connection_chans
                layers = [
                    ResBlock(
                        level_and_skip_chans,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        signal_dim=signal_dim,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                if downsample_rate in attention_resolutions:
                    layers.append(
                        SelfAttentionBlock(
                            dim=level_and_skip_chans,
                            num_heads=num_heads,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                        )
                        if cond_type == "cat"
                        else CrossAttentionBlock(
                            dim=level_and_skip_chans,
                            num_heads=num_heads,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                        )
                    )
                if level > 0:
                    upsampled_chans = input_block_chans[level - 1]
                    layers.append(
                        ResBlock(
                            in_channels=level_and_skip_chans,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=upsampled_chans,
                            signal_dim=signal_dim,
                            use_scale_shift_norm=use_scale_shift_norm,
                            upsample=True,
                        )
                        if resblock_updown
                        else Upsample(
                            in_channels=level_and_skip_chans,
                            use_conv=conv_resample,
                            signal_dim=signal_dim,
                            out_channels=upsampled_chans,
                        )
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out_proj = nn.Sequential(
            normalization(level_and_skip_chans),
            nn.SiLU(),
            zero_module(conv_nd(signal_dim, in_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, cond=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param cond: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (cond is not None) == (
            self.cond_type == "xattn"
        ), "cond should be provided iff cond_type is xattn"

        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        hs = []  # Hidden states for skip connections

        for blk in self.input_blocks:
            x = blk(x, emb, cond)
            hs.append(x)

        x = self.middle_block(x, emb, cond)

        for blk in self.output_blocks:
            x = torch.cat([x, hs.pop()], dim=1)
            x = blk(x, emb, cond)

        return self.out_proj(x)


class PatchedUNetCatCond(UNetModel):
    """Patched UNet with conditioning upsampled and concatenated to input.
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
        cond_channels: int,
        patch_size: int,
        *args,
        **kwargs,
    ):
        in_channels_p = in_channels * patch_size * patch_size + cond_channels
        out_channels_p = out_channels * patch_size * patch_size
        super().__init__(
            in_channels=in_channels_p,
            out_channels=out_channels_p,
            cond_type="cat",
            *args,
            **kwargs,
        )
        self.P_H, self.P_W = patch_size, patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        sample: torch.FloatTensor,  # Shape (B, C, H, W)
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor = None,  # Shape (B, D_C, H_C, W_C)
        # Boolen tensor of shape (B, H_C, W_C). True for masked out pixels
        cond_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        B, C, H, W = sample.shape
        assert (H % self.P_H == 0) and (
            W % self.P_W == 0
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

        # Optionally mask out conditioning
        if cond_mask is not None:
            encoder_hidden_states = torch.where(
                cond_mask[:, None, :, :], 0.0, encoder_hidden_states
            )

        # Concat input with upsampled conditioning
        cond_upsampled = F.interpolate(
            encoder_hidden_states, (N_H, N_W), mode="nearest"
        )
        x = torch.cat([x, cond_upsampled], dim=1)

        # UNet forward pass in subspace
        x = super().forward(x, timestep, **kwargs)

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


class PatchedUNetXattnCond(UNetModel):
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
            cond_type="xattn",
            *args,
            **kwargs,
        )
        self.P_H, self.P_W = patch_size, patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        sample: torch.FloatTensor,  # Shape (B, C, H, W)
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor = None,  # Shape (B, N, D)
        **kwargs,
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

        x = super().forward(x, timestep, cond=cond, **kwargs)

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


def unet_patched_xattn_cond(**kwargs):
    return PatchedUNetXattnCond(
        patch_size=4,
        model_channels=256,
        num_res_blocks=3,
        attention_resolutions=[4, 8],
        channel_mult=(1, 2, 2, 2),
        **kwargs,
    )


def unet_patched_cat_cond(**kwargs):
    return PatchedUNetCatCond(
        patch_size=4,
        model_channels=256,
        num_res_blocks=3,
        attention_resolutions=[4, 8],
        channel_mult=(1, 2, 2, 2),
        **kwargs,
    )
