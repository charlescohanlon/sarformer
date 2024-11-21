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
import random
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import truncnorm

from src.utils import to_2tuple


class AbstractImageAugmenter(ABC):
    """Abstract class for image augmenters."""

    @abstractmethod
    def __call__(self, uid=None):
        pass


class CropImageAugmenter(AbstractImageAugmenter): # TODO: need to work for both finetuning and pretraining

    def __init__(
        self,
        target_size=224,
        crop_std=1.0,
        hflip=None,
        vflip=None,
    ):
        self.target_size = to_2tuple(target_size)
        self.crop_std = crop_std
        self.hflip = hflip
        self.vflip = vflip

    def __call__(self):
        # center distribution at center of valid crop range
        start_idx, end_idx = 0, self.target_size[0] - 1
        mean = (start_idx + end_idx) / 2

        # compute bounds of truncated normal distribution in stds
        lower_std = (start_idx - mean) / self.crop_std
        upper_std = (end_idx - mean) / self.crop_std

        # sample truncated normal distribution and round to discretize it
        crop_row = round(
            truncnorm.rvs(lower_std, upper_std, loc=mean, scale=self.crop_std)
        )
        crop_col = round(
            truncnorm.rvs(lower_std, upper_std, loc=mean, scale=self.crop_std)
        )
        hflip = random.random() < self.hflip if self.hflip is not None else False
        vflip = random.random() < self.vflip if self.vflip is not None else False

        crop_coords = (crop_row, crop_col, *self.target_size)
        flip = (hflip, vflip)

        return crop_coords, flip, None, self.target_size, None


class EmptyAugmenter(AbstractImageAugmenter):
    def __init__(self):
        pass

    def __call__(self, uid=None):
        return None, None, None, None, None
