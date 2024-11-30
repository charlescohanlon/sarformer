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
)
from src.utils import generate_uint15_hash
from pathlib import Path

MODALITY_INFO = {
    "target_distribution": {
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("target_distribution"),
    },
    "mask": {
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("mask"),
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
        "type": "seq",
        "id": generate_uint15_hash("caption"),
        "path": "caption/",
        "index_col": "UID",
    },
    "structured": {
        "type": "seq",
        "id": generate_uint15_hash("structured"),
        "path": "structured/",
        "index_col": "uid",
    },
}

MODALITY_TRANSFORMS = {
    "caption": CaptionTransform(shuffle=True),
    # "structured": StructuredDataTransform(shuffle=True),
    "rgb": RGBTransform(),
    "depth": DepthTransform(norm_ops=["depth_minmax_scaling"]),
}
