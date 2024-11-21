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
from src.data.modality_transforms import (
    CaptionTransform,
    DepthTransform,
    RGBTransform,
    StructuredDataTransform,
    TargetDistributionTransform,
)
from src.utils import generate_uint15_hash
from pathlib import Path

SCRATCH_DIR = Path("/scratch/bdej/cohanlon")
IMG_SIZE = 224

MODALITY_INFO = {
    "target_distribution": {
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("target_distribution"),
    },
    "rgb": {
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("rgb"),
        "path": "rgb/",
    },
    "depth": {
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("depth"),
        "path": "depth/",
    },
    "caption": {
        "min_tokens": 0,
        "max_tokens": 516,
        "type": "seq",
        "id": generate_uint15_hash("caption"),
        "path": "caption/",
    },
    "structured_data": {
        "min_tokens": 0,
        "max_tokens": None,
        "type": "seq",
        "id": generate_uint15_hash("structured_data"),
        "path": "structured/",
    },
}

USED_COLS = {
    "10m_u_component_of_wind": "",
    "10m_v_component_of_wind": "",
    "2m_dewpoint_temperature": "",
    "2m_temperature": "",
    "surface_pressure": "",
    "total_precipitation": "",
    "total_cloud_cover": "",
    "low_cloud_cover": "",
    "slope_of_subgridscale_orography": "",
    "high_vegetation_cover": "",
    "surface_net_solar_radiation": "",
    "soil_type": "",
    "trapping_layer_base_height": "",
    "total_column_water_vapour": "",
    "skin_temperature": "",
    "precipitation_type": "",
    "min_elevation": "",
    "max_elevation": "",
}

MODALITY_TRANSFORMS = {
    "caption": CaptionTransform(
        return_attn_mask=True,
        shuffle=True,
        root=SCRATCH_DIR / "caption",
        tokenizer_name="t5-small",
        index_col_name="uid",
    ),
    "structured_data": StructuredDataTransform(
        col_map=USED_COLS,
        shuffle=True,
        root=SCRATCH_DIR / "structured_data",
        tokenizer_name="t5-small",
        index_col_name="uid",
    ),
    "target_distribution": TargetDistributionTransform(img_size=IMG_SIZE),
    "rgb": RGBTransform(mean_and_std="naip"),
    "depth": DepthTransform(norm_ops=["depth_minmax_scaling"]),
}
