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

import numpy as np
import pandas as pd

from fourm.data.modality_transforms import (
    CaptionTransform,
    DepthTransform,
    RGBTransform,
    TokTransform,
    StructuredDataTransform,
    TargetDistributionTransform,
)
from fourm.models.decoder_embeddings import (
    ImageTokenDecoderEmbedding,
    SequenceDecoderEmbedding,
)
from fourm.models.encoder_embeddings import (
    ImageTokenEncoderEmbedding,
    SequenceEncoderEmbedding,
)
from fourm.utils import generate_uint15_hash

MODALITY_INFO = {
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
        "path": None,  # path None indicates data comes from csv
    },
    "target_distribution": {
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("target_distribution"),
        "path": None,
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
    "rgb": {  # used for tokenizer training
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("rgb"),
        "path": "rgb",
        "no_data_value": 0,
    },
    "depth": {  # used for tokenizer training
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("depth"),
        "no_data_value": -9999.0,
        "path": "depth",
    },
    "structureddata": {
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
        "max_tokens": None,
        "type": "seq",
        "id": generate_uint15_hash("structureddata"),
        "path": None,
    },
}

ID_MAP = {  # for structured data transform
    k: i
    for i, k in enumerate(
        [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_pressure",
            "total_precipitation",
            "total_cloud_cover",
            "low_cloud_cover",
            "slope_of_subgridscale_orography",
            "high_vegetation_cover",
            "surface_net_solar_radiation",
            "soil_type",
            "trapping_layer_base_height",
            "total_column_water_vapour",
            "skin_temperature",
            "precipitation_type",
            "duration",
            "start_date",
            "start_time",
            "prompt",
            "min_elevation",
            "max_elevation",
        ]
    )
}

MODALITY_TRANSFORMS = {
    "caption": CaptionTransform(caption_name="prompt"),
    "tok_rgb": TokTransform(),
    "tok_depth": TokTransform(),
    "structureddata": StructuredDataTransform(
        id_map=ID_MAP, shuffle=True, value_type=np.float16
    ),
    "target_distribution": TargetDistributionTransform(
        spatial_res=2,
        img_size=224,
        offset_name="offset",
        bearing_name="offset_bearing",
        resize_ratio=224 / 1000,
    ),
}

MODALITY_TRANSFORMS_DIVAE = {
    "rgb": RGBTransform(
        mean_and_std="naip", no_data_value=MODALITY_INFO["rgb"]["no_data_value"]
    ),
    "depth": DepthTransform(
        norm_ops=["depth_minmax_scaling"],
        no_data_value=MODALITY_INFO["depth"]["no_data_value"],
    ),
}
