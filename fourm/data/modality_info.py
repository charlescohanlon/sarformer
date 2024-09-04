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
from functools import partial

import fourm.utils.data_constants as data_constants
from fourm.data.modality_transforms import (
    CaptionTransform,
    DepthTransform,
    DetectionTransform,
    MaskTransform,
    NormalTransform,
    RGBTransform,
    SemsegTransform,
    TokTransform,
    CaptionEmbTransform,
    MetadataTransform,
    HumanPoseTransform,
    ColorPaletteTransform,
    SAMInstanceTokTransform,
    SAMInstanceTransform,
)
from fourm.models.decoder_embeddings import (
    ImageTokenDecoderEmbedding,
    SequenceDecoderEmbedding,
)
from fourm.models.encoder_embeddings import (
    ImageEncoderEmbedding,
    ImageTokenEncoderEmbedding,
    SequenceEncoderEmbedding,
    SequenceEmbEncoderEmbedding,
)
from fourm.utils import generate_uint15_hash

MODALITY_INFO = {
    "rgb": {  # used for tokenizer training
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("rgb"),
        "path": "rgb",
        "no_data_value": 0,
    },
    "caption": {
        "vocab_size": 30_000,
        "encoder_embedding": partial(
            SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0
        ),
        "decoder_embedding": partial(
            SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0
        ),
        "min_tokens": 0,
        "max_tokens": 256,
        "type": "seq",
        "id": generate_uint15_hash("caption"),
    },
    "tok_rgb@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 16384,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=16384),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=16384),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_rgb@224"),
        "pretokenized": True,
    },
    "tok_depth@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8192,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_depth@224"),
        "pretokenized": True,
    },
    "depth": {  # used for tokenizer training
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("depth"),
        "no_data_value": -9999.0,
    },
    "metadata": {
        "vocab_size": 30_000,
        "encoder_embedding": partial(
            SequenceEncoderEmbedding,
            vocab_size=30_000,
            max_length=40,
            padding_idx=0,
            sincos_pos_emb=True,
        ),
        "decoder_embedding": partial(
            SequenceDecoderEmbedding,
            vocab_size=30_000,
            max_length=40,
            padding_idx=0,
            sincos_pos_emb=True,
        ),
        "min_tokens": 0,
        "max_tokens": 40,  # At most 2x19=38 for 19 metadata types, +1 for EOS, +1 for sentinel
        "type": "seq",
        "id": generate_uint15_hash("metadata"),
        "shared_vocab": ["caption"],
        "path": "metadata",
    },
}

# Note: @res suffix is ignored for modality transforms
MODALITY_TRANSFORMS = {
    # 4M-7 modalities
    "rgb": None,
    "caption": CaptionTransform(aligned_captions=True),
    "det": DetectionTransform(
        det_threshold=0.6,
        det_max_instances=None,
        bbox_order="dist_to_orig",
        coord_bins=1000,
        min_visibility=0.0,
    ),
    "tok_rgb": TokTransform(),
    "tok_depth": TokTransform(),
    "tok_normal": TokTransform(),
    "tok_semseg": TokTransform(),
    "tok_clip": TokTransform(),
    # 4M-21 modalities
    "t5_caption": CaptionEmbTransform(),
    "metadata": MetadataTransform(
        special_vmin=0,
        special_vmax=999,
        shuffle=True,
        random_trunc=False,
        return_chunks=True,
    ),
    "human_poses": HumanPoseTransform(coord_bins=1000),
    "color_palette": ColorPaletteTransform(coord_bins=1000),
    "sam_instance": SAMInstanceTokTransform(
        image_size=224, points_per_side=7, point_order="random"
    ),
    "tok_canny_edge": TokTransform(),
    "tok_sam_edge": TokTransform(),
    "tok_dinov2": TokTransform(),
    "tok_imagebind": TokTransform(),
    "tok_dinov2_global": TokTransform(),
    "tok_imagebind_global": TokTransform(),
    # Other
    "mask_valid": MaskTransform(mask_pool_size=1),
}

MODALITY_TRANSFORMS_DIVAE = {
    "rgb": RGBTransform(
        mean_and_std="naip", no_data_value=MODALITY_INFO["rgb"]["no_data_value"]
    ),
    "depth": DepthTransform(
        norm_ops=["depth_minmax_scaling"],
        no_data_value=MODALITY_INFO["depth"]["no_data_value"],
    ),
    "normal": NormalTransform(standardize_surface_normals=False),
    "mask_valid": MaskTransform(mask_pool_size=1),
    "semseg_coco": SemsegTransform(shift_idx_by_one=True),
    "canny_edge": RGBTransform(imagenet_default_mean_and_std=False),
    "human_poses": HumanPoseTransform(coord_bins=1000, only_pose=True),
    "sam_mask": SAMInstanceTransform(mask_size=64, max_instance_n=1),
}

MODALITY_TRANSFORMS_VQCONTROLNET = {
    "rgb": RGBTransform(imagenet_default_mean_and_std=False),
    "mask_valid": MaskTransform(mask_pool_size=1),
    "caption": CaptionTransform(aligned_captions=True),
}
