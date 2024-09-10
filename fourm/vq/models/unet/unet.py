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
from typing import Dict, Union, Optional

import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fourm.models.fm_utils import softmax1

from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

from .fp16_util import (
    convert_module_to_f16,
    convert_module_to_f32,
    convert_module_to_bf16,
)
from .nn import (
    checkpoint,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


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

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

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

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
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
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
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
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        B, C, *spatial = x.shape
        x = x.reshape(B, C, -1)
        # TODO: should be pre and post norm (also this pre norm is out of order)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(B, C, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "b c t, b c s -> b t s",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "b t s, b c s -> b c t", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.use_checkpoint = use_checkpoint

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        return checkpoint(
            self._forward, (x, cond), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, cond):
        B, N, C = x.shape
        # B, N, C = x.shape
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
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = softmax1(attn, dim=-1)  # NOTE: we're using the custom softmax here
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class UNetModel(ModelMixin, ConfigMixin):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param cond_type: the method used to apply conditioning (either 'cat' or 'xattn').
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size=224,
        in_channels=3,
        model_channels=256,
        out_channels=3,
        num_res_blocks=3,
        attention_resolutions=[8, 16],
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        cond_type="cat",
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        if cond_type not in ("cat", "xattn"):
            raise ValueError(f"Unknown cond_type {cond_type}")
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.sample_size = image_size  # For compatibility
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.cond_type = cond_type
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        C = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, C, 3, padding=1))]
        )
        self._feature_size = C
        input_block_chans = [C]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        C,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                C = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            C,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += C
                input_block_chans.append(C)
            if level != len(channel_mult) - 1:
                out_ch = C
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            C,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            C, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                C = out_ch
                input_block_chans.append(C)
                ds *= 2
                self._feature_size += C

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                C,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    C,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    use_new_attention_order=use_new_attention_order,
                )
                if self.cond_type == "cat"
                else CrossAttentionBlock(C, num_heads=num_heads)
            ),
            ResBlock(
                C,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += C

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                in_chans = input_block_chans.pop()
                layers = [
                    ResBlock(
                        C + in_chans,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                C = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            C,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = C
                    layers.append(
                        ResBlock(
                            C,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(C, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += C

        self.out_proj = nn.Sequential(
            normalization(C),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
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
            x = blk(x, emb)
            hs.append(x)

        if self.cond_type == "xattn":
            x = self.middle_block(x, emb, cond)
        else:
            x = self.middle_block(x, emb)

        for blk in self.output_blocks:
            x = torch.cat([x, hs.pop()], dim=1)
            x = blk(x, emb)

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
            in_channels=in_channels_p, out_channels=out_channels_p, *args, **kwargs
        )
        self.P_H, self.P_W = pair(patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        sample: th.FloatTensor,  # Shape (B, C, H, W)
        timestep: Union[th.Tensor, float, int],
        encoder_hidden_states: th.Tensor = None,  # Shape (B, D_C, H_C, W_C)
        # Boolen tensor of shape (B, H_C, W_C). True for masked out pixels
        cond_mask: Optional[th.BoolTensor] = None,
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
        x = th.cat([x, cond_upsampled], dim=1)

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
        cond_channels: int,
        patch_size: int,
        *args,
        **kwargs,
    ):
        in_channels_p = in_channels * patch_size * patch_size + cond_channels
        out_channels_p = out_channels * patch_size * patch_size
        super().__init__(
            in_channels=in_channels_p, out_channels=out_channels_p, *args, **kwargs
        )
        self.P_H, self.P_W = pair(patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        sample: torch.FloatTensor,  # Shape (B, C, H, W)
        timestep: Union[th.Tensor, float, int],
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
