import math
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
from einops import repeat
from torch import nn

from fourm.utils.timm.registry import register_model
from fourm.vq.scheduling.diffusion_pipeline import PipelineCond
from fourm.vq.scheduling.scheduling_ddim import DDIMScheduler
from fourm.vq.scheduling.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from .fm_utils import EncoderBlock, LayerNorm
from fourm.models.unet import PatchedConvNeXtUNet
from fourm.data.modality_info import MODALITY_INFO

# Model definitions
__all__ = ["fm_b_12e_patched_convnext_unet_b_swiglu_qknorm_nobias"]


class SARFormer(nn.Module):
    """SARFormer model.

    Args:
        encoder_embeddings: Dict of encoder embedding modules.
        modality_info: Dict containing modality information.
        dim: Embedding dimension.
        encoder_depth: Number of encoder blocks.
        num_heads_encoder: Number of attention heads in the encoder.
        num_heads_backbone: Number of attention heads in the backbone.
        mlp_ratio_encoder: Ratio of mlp hidden dim to embedding dim.
        mlp_ratio_backbone: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value projections.
        proj_bias: If True, add a learnable bias to the last projection of the attention block.
        mlp_bias: If True, add a learnable bias to linear layers in the MLP / feed-forward.
        drop_path_rate_encoder: Stochastic depth rate for encoder.
        patch_size_backbone: Patch size for the backbone.
        num_conv_blocks_backbone: Number of convolutional blocks per level in the backbone.
        num_channels_backbone: Number of channels at the first level of the backbone.
        channel_mult_backbone: Channel multiplier for each level of the backbone.
        drop_path_rate_backbone: Stochastic depth rate for backbone.
        act_layer: Activation layer to be used.
        norm_layer: Normalization layer to be used.
        gated_mlp: If True, make the feedforward gated (e.g., SwiGLU).
        qk_norm: If True, applies normalization to queries and keys (QKNorm).
        scheduler_type: String identifier specifying the diffusion scheduler to use.
            Can be 'ddpm' or 'ddim'.
        num_train_timesteps: Number of diffusion timesteps to use for training.
        thresholding: Whether or not to use dynamic thresholding  (introduced by Imagen,
            https://arxiv.org/abs/2205.11487) for the diffusion process, at inference only.
        beta_schedule: String identifier specifying the beta schedule to use for
            the diffusion process. Can be 'linear', 'squaredcos_cap_v2' (cosine),
            'shifted_cosine:{shift_amount}'; see vq/scheduling for details.
        prediction_type: String identifier specifying the type of prediction to use.
            Can be 'sample', 'epsilon', or 'v_prediction'; see vq/scheduling for details.
        zero_terminal_snr: Whether or not to enforce zero terminal SNR, i.e. the SNR
            at the last timestep is set to zero. This is useful for preventing the model
            from "cheating" by using information in the last timestep to reconstruct the image.
            See https://arxiv.org/abs/2305.08891.
    """

    def __init__(
        self,
        encoder_embeddings: Dict[str, nn.Module],
        modality_info: Dict[str, Any],
        dim: int = 768,
        encoder_depth: int = 12,
        num_heads_encoder: int = 12,
        num_heads_backbone: int = 8,
        mlp_ratio_encoder: float = 4.0,
        mlp_ratio_backbone: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        drop_path_rate_encoder: float = 0.0,
        patch_size_backbone: int = 4,
        num_conv_blocks_backbone: int = 4,
        num_channels_backbone: int = 128,
        channel_mult_backbone: Tuple[int, ...] = (1, 2, 4, 8, 16),
        drop_path_rate_backbone: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: Union[partial, nn.Module] = partial(LayerNorm, eps=1e-6),
        gated_mlp: bool = False,  # Make the feedforward gated for e.g. SwiGLU
        qk_norm: bool = False,
        scheduler_type: str = "ddpm",
        num_train_timesteps: int = 1000,
        thresholding: bool = False,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "sample",
        zero_terminal_snr: bool = True,
    ):
        super().__init__()

        self.modality_info = modality_info
        self.dim = dim
        self.init_std = 0.02

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

        self.backbone = PatchedConvNeXtUNet(
            in_channels=1,
            out_channels=1,
            cond_dim=dim,
            patch_size=patch_size_backbone,
            num_conv_blocks=num_conv_blocks_backbone,
            model_channels=num_channels_backbone,
            channel_mult=channel_mult_backbone,
            drop_path_rate=drop_path_rate_backbone,
            num_heads=num_heads_backbone,
            mlp_ratio=mlp_ratio_backbone,
            qkv_bias=qkv_bias,
            act_layer=act_layer,
        )

        # Init diffusion scheduler / default pipeline for generation
        scheduler = DDPMScheduler if scheduler_type == "ddpm" else DDIMScheduler
        self.noise_scheduler = scheduler(
            num_train_timesteps=num_train_timesteps,
            thresholding=thresholding,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            zero_terminal_snr=zero_terminal_snr,
        )
        self.pipeline = PipelineCond(model=self, scheduler=self.noise_scheduler)

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

    def get_num_encoder_params(self):
        return sum(p.numel() for p in self.encoder.parameters())

    def get_num_backbone_params(self):
        return sum(p.numel() for p in self.backbone.parameters())

    def get_num_params(self):
        return self.get_num_encoder_params() + self.get_num_backbone_params()

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

    def _get_pipeline(self, scheduler: Optional[SchedulerMixin] = None) -> PipelineCond:
        """Creates a conditional diffusion pipeline with the given scheduler.

        Args:
            scheduler: Scheduler to use for the diffusion pipeline.
              If None, the default scheduler will be used.

        Returns:
            Conditional diffusion pipeline.
        """
        return (
            PipelineCond(model=self, scheduler=scheduler)
            if scheduler is not None
            else self.pipeline
        )

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
                - encoder_tokens_all (torch.Tensor): Concatenated encoder tokens from all modalities. Shape (B, N, D) where N is the total number of all encoder tokens.
                - emb_all (torch.Tensor): Concatenated encoder embeddings from all modalities. Shape (B, N, D)
                - encoder_mask_all (torch.Tensor): Mainly to mask the [PAD] tokens in the encoder. Shape (B, N)
        """

        encoder_tokens_all = []
        emb_all = []
        encoder_mask_all = []

        for d in mod_dict.values():
            if isinstance(d["x"], tuple):
                x, mask = d["x"]
                encoder_tokens_all.append(x)
                encoder_mask_all.append(mask.to(torch.bool, non_blocking=True))
            else:
                encoder_tokens_all.append(d["x"])
                encoder_mask_all.append(torch.ones_like(d["x"], dtype=torch.bool))

            emb_all.append(d["emb"])

        encoder_tokens_all = torch.cat(encoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)
        encoder_mask_all = torch.cat(encoder_mask_all, dim=1)

        return encoder_tokens_all, emb_all, encoder_mask_all

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
                - encoder_mask (torch.Tensor): Mainly to mask the [PAD] tokens in the encoder. Shape (B, N)

        """
        encoder_tokens_all, emb_all, encoder_mask_all = self.cat_encoder_tensors(
            mod_dict
        )

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
        encoder_mask = torch.gather(
            encoder_mask_all,
            dim=1,
            index=ids_shuffle,
        )

        return encoder_tokens, encoder_emb, encoder_mask

    def forward_encoder(
        self, cond_mod_dict: Union[Dict[str, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        """Forward pass for the encoder from the mod_dict.

        Args:
            cond_mod_dict (dict): Dictionary containing tensors for the different conditioning
                modalities. It is expected to have keys for each modality and values containing
                the modalities' associated tensors.

        Returns:
            torch.Tensor: Encoder output. Shape (B, N, D)
        """
        encoder_mod_dict = {
            mod: self.encoder_embeddings[mod](d) for mod, d in cond_mod_dict.items()
        }
        encoder_tokens, encoder_emb, encoder_mask = self.prep_encoder_tokens(
            encoder_mod_dict
        )

        x = encoder_tokens + encoder_emb
        for blk in self.encoder:
            x = blk(x, mask=encoder_mask)

        x = self.encoder_norm(x)
        context = self.context_proj(x) + encoder_emb

        return context

    def forward(
        self,
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
        cond: Union[Dict[str, Dict[str, torch.Tensor]], torch.Tensor],
        unconditional: bool = False,
        **kwargs,  # to absorb additional arguments from PipelineCond
    ) -> torch.Tensor:
        """Forward pass of the SARFormer model.

        Args:
            noisy_image (torch.Tensor): Noisy image tensor. Shape (B, C, H, W)
            timesteps (torch.Tensor): Timestep tensor. Shape (B)
            cond_mod_dict (dict): Dictionary containing tensors for the different conditioning
                modalities. It is expected to have keys for each modality and values containing
                the modalities' associated tensors.
            unconditional (bool): If True, the model is run in unconditionally.

        Returns:
            torch.Tensor: The scalar field logits. Shape (B, 1, H, W)

        """
        if isinstance(cond, dict):
            context = self.forward_encoder(cond)
        else:
            context = cond

        # Diffusion Backbone
        x = self.backbone(noisy_image, timesteps, context)

        return x

    def generate(
        self,
        cond_mod_dict: Dict[str, Dict[str, torch.Tensor]],
        timesteps: Optional[int] = None,
        scheduler: Optional[SchedulerMixin] = None,
        generator: Optional[torch.Generator] = None,
        cfg_scale: float = 0.0,
        image_size: Optional[Union[Tuple[int, int], int]] = None,
        verbose: bool = False,
        scheduler_timesteps_mode: str = "trailing",
    ) -> torch.Tensor:
        """Decodes quantized latent codes back to an image.

        Args:
            cond_mod_dict (dict): Dictionary containing tensors for the different conditioning
                modalities. It is expected to have keys for each modality and values containing
                the modalities' associated tensors.
            timesteps: Number of diffusion timesteps to use. Defaults to self.num_train_timesteps.
            scheduler: Scheduler to use for the diffusion pipeline. Defaults to the training scheduler.
            generator: Random number generator to use for sampling. By default generations are stochastic.
            cfg_scale: The scale of the classifier-free guidance. If set to 0.0, no guidance is used.
            image_size: Image size to use for the diffusion pipeline. Defaults to decoder image size.
            verbose: Whether or not to print progress bar.
            scheduler_timesteps_mode: The mode to use for DDIMScheduler. One of `trailing`, `linspace`,
                `leading`. See https://arxiv.org/abs/2305.08891 for more details.

        Returns:
            Decoded image tensor of shape B C H W
        """
        cond = self.forward_encoder(cond_mod_dict)
        pipeline = self._get_pipeline(scheduler)
        result = pipeline(
            cond,
            timesteps=timesteps,
            generator=generator,
            guidance_scale=cfg_scale,
            image_size=image_size,
            verbose=verbose,
            scheduler_timesteps_mode=scheduler_timesteps_mode,
        )
        return result


@register_model
def sarformer_t_swiglu_qknorm_nobias(
    encoder_embeddings: Dict[str, nn.Module],
    **kwargs,
):
    model = SARFormer(
        encoder_embeddings=encoder_embeddings,
        modality_info=MODALITY_INFO,
        dim=768,
        encoder_depth=3,
        num_heads_encoder=8,
        num_heads_backbone=8,
        patch_size_backbone=4,
        num_channels_backbone=128,
        num_conv_blocks_backbone=3,
        channel_mult_backbone=(1, 2, 4, 8, 16),
        mlp_ratio_encoder=4.0,
        mlp_ratio_backbone=4.0,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        qk_norm=True,
        **kwargs,
    )
    return model
