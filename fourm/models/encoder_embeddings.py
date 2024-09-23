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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .fm_utils import build_1d_sincos_posemb, build_2d_sincos_posemb, pair


class SequenceEncoderEmbedding(nn.Module):
    """Embedding module for encoding sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        max_sincos_pos_emb: Maximum allowed length for sin-cos positional embeddings
        padding_idx: Padding index for word embedding
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim_tokens: Optional[int] = None,
        sincos_pos_emb: bool = True,
        max_sincos_pos_emb: int = 512,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(
                    f"Max length ({self.max_length}) is greater than the number of posembs ({self.max_sincos_pos_emb}"
                )
            pos_emb = build_1d_sincos_posemb(
                max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens
            )[: self.max_length]
            # self.pos_emb is now a buffer for FSDP
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.max_length, self.dim_tokens)
            )
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.dim_tokens,
            padding_idx=self.padding_idx,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings

        Args:
            x (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): If a tensor, x is the Input
                token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence
                length. if a tuple, x is the input token sequence and mask. Shape (B, L) and (B, L) respectively.

        Returns:
        Tuple:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
                - mask: (torch.Tensor): Attention mask for the input sequence. Shape (B, L).
        """
        mask = None
        if isinstance(x, tuple):
            ids, mask = x
        else:
            ids = x

        B = ids.shape[0]

        # Map to embedding
        x = self.token_emb(ids)

        x_emb = repeat(self.pos_emb + self.mod_emb, "() n d -> b n d", b=B)

        # Create positional embedding + modality embedding
        return x, x_emb, mask


class ImageTokenEncoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        dim_tokens: Optional[int] = None,
        sincos_pos_emb: bool = True,
        image_size: Union[int, Tuple[int]] = 224,
        **kwargs,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (
            self.image_size[1] // patch_size
        )

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of module that are dependent on dimension of tokens.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(
                h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens
            )
            # self.pos_emb is now a buffer for FSDP
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens)
            )
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(
        self, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through embedding module, transforming image tokens to a sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            ids (torch.Tensor): Input image tensor. Shape (B, 1, H*W)

        Returns:
            Tuple:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, H*W, D).
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, H*W, D).
                - None: No attention mask for image tokens.
        """
        B = ids.shape[0]

        ids = ids.squeeze(1) # (B, 1, H*W) -> (B, H*W)

        # Map to embedding
        x = self.token_emb(ids)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, "() n d -> b n d", b=B)

        return x, x_emb, None  # None for no mask
