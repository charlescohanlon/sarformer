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
from functools import reduce
import json
import random
from typing import Optional, Tuple, Union, Dict
from abc import ABC, abstractmethod

from PIL import Image
from math import radians, sin, cos
import struct

from scipy.stats import iqr
import numpy as np
from tokenizers import Tokenizer
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import pandas as pd

from fourm.data.image_augmenter import AbstractImageAugmenter
from fourm.utils import to_2tuple
from fourm.utils.data_constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    NAIP_MEAN,
    NAIP_STD,
)


# The @-symbol is used to specify the resolution of a modality. Syntax: modality@resolution
def get_transform_key(mod_name):
    return mod_name.split("@")[0] if "@" in mod_name else mod_name


def get_transform_resolution(mod_name, default_resolution, to_tuple=True):
    res = int(mod_name.split("@")[1]) if "@" in mod_name else default_resolution
    return to_2tuple(res) if to_tuple else res


def get_transform(mod_name, transforms_dict):
    return transforms_dict.get(get_transform_key(mod_name), IdentityTransform())


def get_pil_resample_mode(resample_mode: str):
    """
    Returns the PIL resampling mode for the given resample mode string.

    Args:
        resample_mode: Resampling mode string
    """
    if resample_mode is None:
        return None
    elif resample_mode == "bilinear":
        return (
            Image.Resampling.BILINEAR
            if hasattr(Image, "Resampling")
            else Image.BILINEAR
        )
    elif resample_mode == "bicubic":
        return (
            Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        )
    elif resample_mode == "nearest":
        return (
            Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        )
    else:
        raise ValueError(f"Resample mode {resample_mode} is not supported.")


class UnifiedDataTransform(object):
    def __init__(
        self,
        transforms_dict,
        image_augmenter: Optional[AbstractImageAugmenter] = None,
        resample_mode: Optional[str] = None,
        **kwargs,
    ):
        """Unified data augmentation for FourM

        Args:
            transforms_dict (dict): Dict of transforms for each modality
            image_augmenter (AbstractImageAugmenter): Image augmenter
            resample_mode (str, optional): Resampling mode for PIL images (default: None -> uses default resampling mode for data type)
                One out of ["bilinear", "bicubic", "nearest", None].
            add_sizes (bool, optional): Whether to add crop coordinates and original size to the output dict
        """

        self.transforms_dict = transforms_dict
        self.image_augmenter = image_augmenter
        self.resample_mode = resample_mode

    def unified_image_augment(self, mod_dict, crop_settings):
        """Apply the image augmenter to all modalities where it is applicable

        Args:
            mod_dict (dict): Dict of modalities
            crop_settings (dict): Crop settings

        Returns:
            dict: Transformed dict of modalities
        """

        if self.image_augmenter is not None:
            crop_coords, flip, orig_size, target_size, rand_aug_idx, rotation_angle = (
                self.image_augmenter(mod_dict, crop_settings)
            )
        else:
            crop_coords, flip, orig_size, target_size, rand_aug_idx, rotation_angle = (
                None,
                False,
                None,
                None,
                None,
                0,
            )

        mod_dict = {
            k: self.transforms_dict[get_transform_key(k)].image_augment(
                v,
                crop_coords=crop_coords,
                flip=flip,
                rotation_angle=rotation_angle,
                orig_size=orig_size,
                target_size=get_transform_resolution(k, target_size),
                rand_aug_idx=rand_aug_idx,
                resample_mode=self.resample_mode,
            )
            for k, v in mod_dict.items()
        }

        return mod_dict

    def __call__(self, mod_dict):
        """Apply the augmentation to a dict of modalities (both image based and sequence based modalities)

        Args:
            mod_dict (dict): Dict of modalities

        Returns:
            dict: Transformed dict of modalities
        """
        crop_settings = mod_dict.pop("crop_settings", None)

        mod_dict = {
            k: get_transform(k, self.transforms_dict).preprocess(v)
            for k, v in mod_dict.items()
        }

        mod_dict = self.unified_image_augment(mod_dict, crop_settings)

        mod_dict = {
            k: get_transform(k, self.transforms_dict).postprocess(v)
            for k, v in mod_dict.items()
        }

        return mod_dict

    def __repr__(self):
        repr = "(UnifiedDataAugmentation,\n"
        repr += ")"
        return repr


class AbstractTransform(ABC):

    @abstractmethod
    def load(self, sample):
        pass

    @abstractmethod
    def preprocess(self, sample):
        pass

    @abstractmethod
    def image_augment(
        self,
        v,
        crop_coords: Tuple,
        flip: bool,
        rotation_angle: float,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        pass

    @abstractmethod
    def postprocess(self, v):
        pass


class ImageTransform(AbstractTransform):

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        # with open(path, 'rb') as f:
        #     img = Image.open(f)
        img = Image.open(path)
        return img

    @staticmethod
    def image_hflip(img: Image, flip: bool):
        """Crop and resize an image

        :param img: Image to crop and resize
        :param flip: Whether to flip the image
        :return: Flipped image (if flip = True)
        """
        if flip:
            img = TF.hflip(img)
        return img

    @staticmethod
    def image_crop_and_resize(
        img: Image, crop_coords: Tuple, target_size: Tuple, resample_mode: str = None
    ):
        """Crop and resize an image

        :param img: Image to crop and resize
        :param crop_coords: Coordinates of the crop (top, left, h, w)
        :param target_size: Coordinates of the resize (height, width)
        :return: Cropped and resized image
        """

        top, left, h, w = crop_coords
        resize_height, resize_width = target_size
        img = TF.crop(img, top, left, h, w)
        resample_mode = get_pil_resample_mode(resample_mode)
        img = img.resize((resize_height, resize_width), resample=resample_mode)
        return img

    @staticmethod
    def image_rotate(
        img: Union[Image.Image, torch.Tensor],
        angle: float,
        fill_val: Union[float, bool],
        resample_mode: str = "nearest",
    ):
        """Rotate an image

        :param img: Image to rotate
        :param angle: Angle of the rotation
        :param fill_val: Value to fill the area outside the transformed image
        :param resample_mode: mode of interpolation
        :return: Rotated image
        """

        resample_mode = get_pil_resample_mode(resample_mode)
        img = TF.rotate(img, angle=angle, interpolation=resample_mode, fill=fill_val)
        return img


class RGBTransform(ImageTransform):

    def __init__(
        self,
        imagenet_default_mean_and_std=True,
        mean_and_std="naip",
        color_jitter=False,
        color_jitter_strength=0.5,
        no_data_value=0,
    ):
        if mean_and_std == "naip":
            self.rgb_mean = NAIP_MEAN
            self.rgb_std = NAIP_STD
        elif mean_and_std == "imagenet_default":
            self.rgb_mean = IMAGENET_DEFAULT_MEAN
            self.rgb_std = IMAGENET_DEFAULT_STD
        elif mean_and_std == "imagenet_inception":
            self.rgb_mean = IMAGENET_INCEPTION_MEAN
            self.rgb_std = IMAGENET_INCEPTION_STD
        else:
            raise ValueError(f"Invalid mean_and_std: {mean_and_std}")

        self.color_jitter = color_jitter
        self.color_jitter_transform = self.random_color_jitter(color_jitter_strength)
        self.no_data_value = no_data_value

    def random_color_jitter(self, strength=0.5):
        # Color Jitter from Pix2Seq and SimCLR
        # Source: https://github.com/google-research/pix2seq/blob/main/data/data_utils.py#L114
        t = T.Compose(
            [
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.8 * strength,
                            contrast=0.8 * strength,
                            saturation=0.8 * strength,
                            hue=0.2 * strength,
                        )
                    ],
                    p=0.8,
                ),
                T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.2),
            ]
        )

        return t

    def rgb_to_tensor(self, img):
        # to_tensor converts to float, rescales to [0, 1], and reshapes to C x H x W
        img = TF.to_tensor(img)
        return img

    def rgb_tensor_norm(self, img):
        img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
        return img

    def load(self, path):
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        sample = sample.convert("RGB")

        if self.color_jitter:
            sample = self.color_jitter_transform(sample)

        return sample

    def image_augment(
        self,
        img,
        crop_coords: Tuple,
        flip: bool,
        rotation_angle: float,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        img = self.image_rotate(img, rotation_angle, self.no_data_value, resample_mode)
        return img

    def postprocess(self, sample):
        sample = self.rgb_to_tensor(sample)
        return sample


class DepthTransform(ImageTransform):

    def __init__(
        self,
        norm_ops=[],
        no_data_value=-9999.0,
    ):
        self.norm_ops = norm_ops
        self.no_data_value = no_data_value

    def depth_to_tensor(self, img):
        # Why? (can't use this with the no_data_value anyways)
        # img = torch.Tensor(img / (2**16 - 1.0))
        img = torch.Tensor(img)
        img = img.unsqueeze(0)  # 1 x H x W
        # if self.standardize_depth:
        #     img = self.truncated_depth_standardization(img)
        return img

    def depth_tensor_norm(self, img):
        for op_str in self.norm_ops:
            if not hasattr(DepthTransform, op_str):
                raise ValueError(f"Invalid depth norm operation: {op_str}")
            img = getattr(DepthTransform, op_str)(img, no_data_value=self.no_data_value)

        return img

    @staticmethod
    def depth_robust_scaling(depth, no_data_value=-9999.0):
        """Depth robust scaling

        :param depth: Depth map
        :param no_data_value: The value to be treated as no data
        :return: Robustly scaled depth map
        """
        # Remove no-data values and nans
        removal_mask = (depth != no_data_value).logical_and(np.isfinite(depth))
        valid_vals = depth[removal_mask]

        return (depth - np.median(valid_vals)) / (iqr(valid_vals) + 1e-6)

    @staticmethod
    def depth_minmax_scaling(depth, no_data_value=-9999.0):
        """Depth relative normalization

        :param depth: Depth map
        :param no_data_value: The value to be treated as no data
        :return: Relative normalized depth map
        """
        # Remove no-data values and nans
        removal_mask = (depth != no_data_value).logical_and(np.isfinite(depth))
        valid_vals = depth[removal_mask]

        min_val = valid_vals.min()
        max_val = valid_vals.max()

        return (depth - min_val) / (max_val - min_val + 1e-6)

    @staticmethod
    def truncated_depth_standardization(
        depth, thresh: float = 0.0, no_data_value=-9999.0
    ):
        """Truncated depth standardization

        :param depth: Depth map
        :param thresh: Threshold
        :param no_data_value: the value to be treated as no data
        :return: Robustly standardized depth map
        """
        # Flatten depth and remove bottom and top 10% of values
        trunc_depth = torch.sort(depth.reshape(-1), dim=0)[0]

        # Remove no-data values and nans
        removal_mask = (depth != no_data_value).logical_and(np.isfinite(depth))
        trunc_depth = trunc_depth[removal_mask]

        trunc_depth = trunc_depth[
            int(thresh * trunc_depth.shape[0]) : int(
                (1 - thresh) * trunc_depth.shape[0]
            )
        ]
        return (depth - trunc_depth.mean()) / torch.sqrt(trunc_depth.var() + 1e-6)

    @staticmethod
    def depth_artifact_mask(depth, no_data_value=-9999.0, outlier_threshold=16):
        """Depth artifact masking
        For masking artifacts that result from clipping of DEM tiffs (essentially masking extreme outliers)

        :param depth: Depth map
        :param no_data_value: The value to be treated as no data
        :param outlier_threshold: The threshold for outlier removal (distance from median in IQRs)
        :return: Depth artifact mask
        """
        # to avoid inflating the median and IQR with values that will be hidden anyways
        removal_mask = (depth != no_data_value).logical_and(np.isfinite(depth))
        filtered_vals = depth[removal_mask]

        # inspired by robust scaling
        dists_from_median = np.abs(depth - np.median(filtered_vals))
        dists_in_iqrs = dists_from_median / (iqr(filtered_vals) + 1e-6)

        # return mask w/ True to keep, False to remove
        return dists_in_iqrs < outlier_threshold

    def load(self, path):
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        img,
        crop_coords: Tuple,
        flip: bool,
        rotation_angle: float,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        img = self.image_rotate(img, rotation_angle, self.no_data_value, resample_mode)
        return img

    def postprocess(self, sample):
        sample = np.array(sample)
        sample = self.depth_to_tensor(sample)
        return sample


class TokTransform(AbstractTransform):

    def __init__(self):
        pass

    def load(self, path):
        sample = np.load(path).astype(int)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        val,
        crop_coords: Tuple,
        flip: bool,
        rotation_angle: float,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        return val

    def postprocess(self, sample):
        return torch.as_tensor(sample)


class CaptionTransform(AbstractTransform):

    def __init__(self, caption_name: str = "prompt"):
        self.caption_name = caption_name

    def set_tokenizer(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def load(self, data: pd.Series):
        return data[self.caption_name]

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        val,
        crop_coords: Tuple,
        flip: bool,
        rotation_angle: float,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        return val

    def postprocess(self, sample):
        assert self.tokenizer is not None, "Tokenizer must be set for caption transform"
        tokens = self.tokenizer.encode(sample)
        return tokens


# TODO: will need to change this when we use a learned tokenizer for the structured data
class StructuredDataTransform(AbstractTransform):

    def __init__(
        self,
        id_map: Dict[str, int],
        shuffle: bool = True,
        value_type: type = np.float16,
    ):
        self.id_map = id_map
        self.shuffle = shuffle
        self.value_type = value_type

    def data_to_str(self, data: pd.Series):
        keys = self.id_map.keys()

        if self.shuffle:
            random.shuffle(keys, len(keys))

        data_str = ""
        for k in keys:
            type_str = f"v0={self.id_map[k]}"

            val = data.astype(self.value_type)[k]
            # Converts floating point to hexadecimal string
            # Courtesy of https://stackoverflow.com/questions/23624212/how-to-convert-a-float-into-hex
            hex_str = str(hex(struct.unpack("<I", struct.pack("<f", val))[0]))[2:]
            hex_str = hex_str.upper()

            byte_strs = [hex_str[i : i + 2] for i in range(0, len(hex_str), 2)]
            val_str = f"v1=[{']['.join(byte_strs)}]"  # e.g. v1=[0A][B3] for float16
            data_str += f"{type_str} {val_str} "

        return data_str

    def set_tokenizer(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def load(self, data: pd.Series):
        return self.data_to_str(data)

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        val,
        crop_coords: Tuple,
        flip: bool,
        rotation_angle: float,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx=None,
        resample_mode: str = None,
    ):
        return val

    def postprocess(self, sample):
        assert (
            self.tokenizer is not None
        ), "Tokenizer must be set for structured data transform"
        tokens = self.tokenizer.encode(sample)
        return tokens


class TargetDistributionTransform(AbstractTransform):

    def __init__(
        self,
        spatial_res: int,  # meters per pixel
        img_size: int,
        offset_name: str = "offset",
        bearing_name: str = "offset_bearing",
        resize_ratio: float = None,
    ):
        self.spatial_res = spatial_res
        self.img_size = img_size
        self.offset_name = offset_name
        self.bearing_name = bearing_name
        self.resize_ratio = resize_ratio

    def load(self, data: pd.Series):
        original_angle = data[self.bearing_name]
        offset = data[self.offset_name] * 1000  # in meters

        # to reverse offset first reverse direction
        reversed_angle = original_angle - 180

        offset_x = offset * cos(radians(reversed_angle))
        offset_y = offset * sin(radians(reversed_angle))

        offset_x_pix = offset_x / self.spatial_res
        offset_y_pix = offset_y / self.spatial_res

        if self.resize_ratio is not None:
            offset_x_pix *= self.resize_ratio
            offset_y_pix *= self.resize_ratio

        center_pix = self.img_size // 2
        x = center_pix + offset_x_pix
        y = center_pix + offset_y_pix

        target_dist = np.zeros((self.img_size, self.img_size))
        target_dist[int(y), int(x)] = 1

        return target_dist

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        val,
        crop_coords: Tuple,
        flip: bool,
        rotation_angle: float,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx=None,
        resample_mode: str = None,
    ):
        # can't do anything b/c the conditioning tokens are already tokenized
        return val

    def postprocess(self, target_distribution):
        return torch.as_tensor(target_distribution)
