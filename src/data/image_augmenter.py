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


class CropImageAugmenter(AbstractImageAugmenter):

    def __init__(
        self,
        img_size,
        eff_img_size=None,
        target_size=224,
        random_crop_std=1e12,
        hflip=0.0,
        vflip=0.0,
    ):
        """
        Args:
            img_size: full size of the input image
            eff_img_size: size of the effective image. Equivalent to taking a center crop
                of full image before cropping. If None, defaults to img_size.
            target_size: size of the output image after cropping
            random_crop_std: standard deviation of the normal distribution used for random cropping.
                Default of 1e12 results in near uniform sampling.
            hflip: probability of horizontal flipping
            vflip: probability of vertical flipping
        """
        self.img_size = img_size
        self.eff_img_size = img_size if eff_img_size is None else eff_img_size
        self.target_size = target_size
        self.crop_std = random_crop_std
        self.hflip = hflip
        self.vflip = vflip

    def __call__(self):
        # compute valid crop range [start, end]
        start = (self.img_size - self.eff_img_size) // 2
        end = self.img_size - start - self.target_size

        mean = (start + end) / 2

        # compute bounds of truncated normal distribution in stds
        lower_std = (start - mean) / self.crop_std
        upper_std = (end - mean) / self.crop_std

        # sample truncated normal distribution and round to discretize it
        top = round(truncnorm.rvs(lower_std, upper_std, loc=mean, scale=self.crop_std))
        left = round(truncnorm.rvs(lower_std, upper_std, loc=mean, scale=self.crop_std))
        hflip = random.random() < self.hflip
        vflip = random.random() < self.vflip

        crop_coords = (top, left)
        flip = (hflip, vflip)

        return crop_coords, flip, None, self.target_size, None


class EmptyAugmenter(AbstractImageAugmenter):
    def __init__(self):
        pass

    def __call__(self, uid=None):
        return None, None, None, None, None
