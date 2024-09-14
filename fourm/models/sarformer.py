from json import encoder
import math
from functools import partial
from typing import Any, Dict, Tuple, Union

import torch
from einops import repeat
from torch import nn

from fourm.utils.timm.registry import register_model

from .fm_utils import EncoderBlock, LayerNorm
import fourm.models.unet as unet
from fourm.data.modality_info import MODALITY_INFO

# Model definitions
__all__ = ["fm_b_12e_patched_convnext_unet_b_swiglu_qknorm_nobias"]


class SARFormer(nn.Module):
    """SARFormer model.

    Args:
        encoder_embeddings: Dict of encoder embedding modules.
        modality_info: Dict containing modality information.
        channels: Number of diffusion backbone input channels.
        backbone_name: Name of the diffusion backbone.
        dim: Embedding dimension.
        encoder_depth: Number of encoder blocks.
        num_heads_encoder: Number of attention heads in the encoder.
        num_heads_backbone: Number of attention heads in the backbone.
        mlp_ratio_encoder: Ratio of mlp hidden dim to embedding dim. The backbone's (effective)
                           mlp ratio is part of the architecture variant function.
        qkv_bias: If True, add a learnable bias to query, key, value projections.
        proj_bias: If True, add a learnable bias to the last projection of the attention block.
        mlp_bias: If True, add a learnable bias to linear layers in the MLP / feed-forward.
        drop_path_rate_encoder: Stochastic depth rate for encoder.
        drop_path_rate_backbone: Stochastic depth rate for backbone.
        act_layer: Activation layer to be used.
        norm_layer: Normalization layer to be used.
        gated_mlp: If True, make the feedforward gated (e.g., SwiGLU).
        qk_norm: If True, applies normalization to queries and keys (QKNorm).
        use_act_checkpoint: If True, use activation checkpoint for each block.
    """

    def __init__(
        self,
        encoder_embeddings: Dict[str, nn.Module],
        modality_info: Dict[str, Any],
        channels: int = 1,
        backbone_name: str = "patched_convnext_unet_b",
        dim: int = 768,
        encoder_depth: int = 12,
        num_heads_encoder: int = 12,
        num_heads_backbone: int = 8,
        mlp_ratio_encoder: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        drop_path_rate_encoder: float = 0.0,
        drop_path_rate_backbone: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: Union[partial, nn.Module] = partial(LayerNorm, eps=1e-6),
        gated_mlp: bool = False,  # Make the feedforward gated for e.g. SwiGLU
        qk_norm: bool = False,
        use_act_checkpoint: bool = False,  # TODO: add activation checkpointing back to unet
    ):
        super().__init__()
        if not hasattr(unet, backbone_name):
            raise ValueError(f"Backbone {backbone_name} not found in unet module.")

        self.modality_info = modality_info
        self.dim = dim
        self.init_std = 0.02
        self.use_act_checkpoint = use_act_checkpoint

        # Encoder embeddings & init
        self.encoder_modalities = set(encoder_embeddings.keys())
        for emb in encoder_embeddings.values():
            emb.init(dim_tokens=dim, init_std=self.init_std)
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)

        ## Transformer encoder
        dpr_encoder = [
            x.item() for x in torch.linspace(0, drop_path_rate_encoder, encoder_depth)
        ]  # stochastic depth decay rule

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    dim=dim,
                    num_heads=num_heads_encoder,
                    mlp_ratio=mlp_ratio_encoder,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    drop_path=dpr_encoder[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    gated_mlp=gated_mlp,
                    qk_norm=qk_norm,
                )
                for i in range(encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(dim)

        ## UNet backbone
        # Projection of encoder tokens before adding the embeddings again
        self.context_proj = nn.Linear(dim, dim)

        self.backbone = getattr(unet, backbone_name)(
            in_channels=channels,
            out_channels=channels,
            cond_dim=dim,
            drop_path_rate=drop_path_rate_backbone,
            num_heads=num_heads_backbone,
            qkv_bias=qkv_bias,
            act_layer=act_layer,
        )

        # Weight init
        self.init_weights()

    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Skipping tokenizers to avoid reinitializing them
            if "tokenizer" in name:
                continue
            # Linear
            elif isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.init_std)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if ".proj" in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                else:
                    # From ConvNeXt
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def get_num_layers_encoder(self):
        return len(self.encoder)

    def get_num_layers_backbone(self):
        # for ConvNeXtUnet counts the number of ConvNeXt and CrossAttention blocks
        return len(self.backbone)

    def get_num_layers(self):
        return self.get_num_layers_encoder() + self.get_num_layers_backbone()

    @torch.jit.ignore
    def no_weight_decay(self):
        # TODO: which modules should be excluded from weight decay?
        no_wd_set = set()

        for mod, emb_module in self.encoder_embeddings.items():
            if hasattr(emb_module, "no_weight_decay"):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f"encoder_embeddings.{mod}.{name}" for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def cat_encoder_tensors(
        self, mod_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        """Concatenate encoder tensors from different modalities.

        Args:
            mod_dict (dict): A dictionary containing information for each modality.
                             Expected keys for each modality are 'x' (input tokens)
                             and 'emb' (embeddings)

        Returns:
            tuple:
                - encoder_tokens_all (torch.Tensor): Concatenated encoder tokens from all modalities. Shape (B, O, D) where O is the total number of all encoder tokens.
                - emb_all (torch.Tensor): Concatenated encoder embeddings from all modalities. Shape (B, O, D)
        """

        encoder_tokens_all = []
        emb_all = []

        for d in mod_dict.values():
            encoder_tokens_all.append(d["x"])
            emb_all.append(d["emb"])

        encoder_tokens_all = torch.cat(encoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)

        return encoder_tokens_all, emb_all

    def prep_encoder_tokens(
        self, mod_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor]:
        """Prepare encoder tokens for the forward pass by concatenating and shuffling them.

        Args:
            mod_dict (dict): Dictionary containing tensors for different modalities.
                            It is expected to have keys for each modality and values
                            containing the modalities' associated tensors.

        Returns:
            tuple:
                - encoder_tokens (torch.Tensor): The encoder tokens. Shape (B, N, D)
                - encoder_emb (torch.Tensor): The encoder embeddings. Shape (B, N, D)

        """
        encoder_tokens_all, emb_all = self.cat_encoder_tensors(mod_dict)

        # shuffle the encoder tokens
        B, N, _ = encoder_tokens_all.shape
        ids_shuffle = torch.randint(N, size=(B, N)).argsort(dim=1)

        # collect tokens in the order of ids_keep (this is supposedly faster than tensor indexing)
        encoder_tokens = torch.gather(
            encoder_tokens_all,
            dim=1,
            index=repeat(ids_shuffle, "b n -> b n d", d=encoder_tokens_all.shape[2]),
        )
        encoder_emb = torch.gather(
            emb_all,
            dim=1,
            index=repeat(ids_shuffle, "b n -> b n d", d=emb_all.shape[2]),
        )

        return encoder_tokens, encoder_emb

    def forward(
        self,
        mod_dict: Dict[str, Dict[str, torch.Tensor]],
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the SARFormer model.

        Args:
            mod_dict (dict): Dictionary containing tensors for different modalities.
                            It is expected to have keys for each modality and values
                            containing the modalities' associated tensors.
            noisy_image (torch.Tensor): Noisy image tensor. Shape (B, C, H, W)
            timesteps (torch.Tensor): Timestep tensor. Shape (B)

        Returns:
            torch.Tensor: The scalar field logits. Shape (B, 1, H, W)

        """
        encoder_mod_dict = {
            mod: self.encoder_embeddings[mod](d) for mod, d in mod_dict.items()
        }
        encoder_tokens, encoder_emb = self.prep_encoder_tokens(encoder_mod_dict)

        # Encoder
        x = encoder_tokens + encoder_emb
        for blk in self.encoder:
            x = blk(x)

        x = self.encoder_norm(x)

        # Diffusion Backbone
        context = self.context_proj(x) + encoder_emb
        x = self.backbone(noisy_image, timesteps, context)

        return x


@register_model
def fm_b_12e_patched_convnext_unet_b_swiglu_qknorm_nobias(
    encoder_embeddings: Dict[str, nn.Module],
    **kwargs,
):
    model = SARFormer(
        encoder_embeddings=encoder_embeddings,
        modality_info=MODALITY_INFO,
        backbone_name="patched_convnext_unet_b",
        dim=768,
        encoder_depth=12,
        num_heads_encoder=12,
        num_heads_backbone=8,
        mlp_ratio_encoder=4.0,
        qkv_bias=False,  # TODO: why no biases? should use no bias for backbone as well?
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        qk_norm=True,
        **kwargs,
    )
    return model
