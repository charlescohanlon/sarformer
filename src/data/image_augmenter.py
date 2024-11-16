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
        target_size=224,
        crop_std=1.0,
        hflip=None,
        vflip=None,
        parameter_csv=None,
    ):
        self.target_size = to_2tuple(target_size)
        self.crop_std = crop_std
        self.hflip = hflip
        self.vflip = vflip
        self.parameter_csv = parameter_csv

    def __call__(self, uid=None):
        if self.parameter_csv is None:
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
        else:
            if uid is None:
                raise ValueError("uid must be provided when using a parameter csv")
            csv_row = self.parameter_csv.loc[uid]

            crop_row = csv_row["crop_row"]
            crop_col = csv_row["crop_col"]
            hflip = csv_row["hflip"]
            vflip = csv_row["vflip"]

        crop_coords = (crop_row, crop_col, *self.target_size)
        flip = (hflip, vflip)

        return crop_coords, flip, None, self.target_size, None, None


class EmptyAugmenter(AbstractImageAugmenter):
    def __init__(self):
        pass

    def __call__(self, uid=None):
        return None, None, None, None, None, None


class RandomRotationImageAugmenter(AbstractImageAugmenter):
    def __init__(self, near_orthogonal=False):
        self.near_orthogonal = near_orthogonal

    def get_rotation_angle(self):
        # rotate image near orthogonally i.e., close to a multiple of 90 degrees
        if self.near_orthogonal:
            rotation = np.random.choice([0, 90, 180, 270])
            offset = np.random.normal(scale=10)  # the std here is arbitrary
            angle = rotation + offset
        else:
            angle = np.random.rand() * 360

        return angle

    def __call__(self, uid=None):
        rotation_angle = self.get_rotation_angle()
        return None, None, None, None, None, rotation_angle
