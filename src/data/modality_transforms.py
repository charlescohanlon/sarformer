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
from copy import deepcopy
import random
from typing import Optional, Text, Tuple, Union, Dict
from abc import ABC, abstractmethod

from PIL import Image
from math import radians, sin, cos

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
    NAIP_MEAN,
    NAIP_STD,
)

from fourm.data.templates import TEMPLATES


# The @-symbol is used to specify the resolution of a modality. Syntax: modality@resolution
def get_modality_prefix(mod_name):
    return mod_name.split("@")[0] if "@" in mod_name else mod_name


def get_transform_resolution(mod_name, default_resolution, to_tuple=True):
    res = int(mod_name.split("@")[1]) if "@" in mod_name else default_resolution
    return to_2tuple(res) if to_tuple else res


def get_transform(mod_name, transforms_dict):
    return transforms_dict.get(get_modality_prefix(mod_name), IdentityTransform())


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
        image_augmenter: Optional[AbstractImageAugmenter],
        resample_mode: Optional[str] = None,
        **kwargs,
    ):
        """Unified data augmentation for FourM

        Args:
            transforms_dict (dict): Dict of transforms for each modality
            image_augmenter (AbstractImageAugmenter): Image augmenter
            resample_mode (str, optional): Resampling mode for PIL images (default: None -> uses default resampling mode for data type)
                One out of ["bilinear", "bicubic", "nearest", None].
        """

        self.transforms_dict = transforms_dict
        self.image_augmenter = image_augmenter
        self.resample_mode = resample_mode

    def unified_image_augment(self, mod_dict, uid=None):
        """Apply the image augmenter to all modalities where it is applicable

        Args:
            mod_dict (dict): Dict of modalities
            uid (str, optional): Unique identifier for the image augmentation

        Returns:
            dict: Transformed dict of modalities
        """

        crop_coords, flip, orig_size, target_size, rand_aug_idx, rotation_angle = (
            self.image_augmenter(uid)
        )

        mod_dict = {
            k: self.transforms_dict[get_modality_prefix(k)].image_augment(
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

    def __call__(self, mod_dict, uid=None):
        """Apply the augmentation to a dict of modalities (both image based and sequence based modalities)

        Args:
            mod_dict (dict): Dict of modalities
            uid (str, optional): Unique identifier for the image augmentation

        Returns:
            dict: Transformed dict of modalities
        """
        mod_dict = {
            k: get_transform(k, self.transforms_dict).preprocess(v)
            for k, v in mod_dict.items()
        }

        mod_dict = self.unified_image_augment(mod_dict, uid)

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


class TextTokenizedTransform(AbstractTransform):
    @abstractmethod
    def set_tokenizer(self, tokenizer: Tokenizer, max_length: int):
        pass

    @abstractmethod
    def load(self, data: pd.Series, uid: str):
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
    def image_flip(img: Image, flip: bool):
        """Horizontally flip an image

        :param img: Image to crop and resize
        :param flip: Whether to flip the image
        :return: Flipped image (if flip = True)
        """
        hflip, vflip = flip
        if hflip:
            img = TF.hflip(img)
        if vflip:
            img = TF.vflip(img)
        return img

    @staticmethod
    def image_crop(img: Image, crop_coords: Tuple):
        """Crop and resize an image

        :param img: Image to crop and resize
        :param crop_coords: Coordinates of the crop (top, left, h, w)
        :return: Cropped and resized image
        """

        row, col, h, w = crop_coords
        img = TF.crop(img, row, col, h, w)
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
        mean_and_std="naip",
        no_data_value=0,
    ):
        if mean_and_std == "naip":
            self.rgb_mean = NAIP_MEAN
            self.rgb_std = NAIP_STD
        else:
            raise ValueError(f"Invalid mean_and_std: {mean_and_std}")

        self.no_data_value = no_data_value

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
        img = self.image_crop(img, crop_coords)
        img = self.image_flip(img, flip)
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
        img = torch.Tensor(img)
        img = img.unsqueeze(0)  # 1 x H x W
        return img

    # Called in __getitem__()
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
        img = self.image_crop(img, crop_coords)
        img = self.image_flip(img, flip)
        return img

    def postprocess(self, sample):
        sample = np.array(sample)
        sample = self.depth_to_tensor(sample)
        return sample


class TargetDistributionTransform(ImageTransform):

    def __init__(
        self,
        spatial_res: int,  # meters per pixel
        img_size: int,
    ):
        self.spatial_res = spatial_res
        self.img_size = img_size

    def load(self, data: pd.Series):
        x = None  # TODO: use parameter_csv for this
        y = None

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
        return val

    def postprocess(self, target_distribution):
        target_distribution = torch.as_tensor(target_distribution)
        return target_distribution.unsqueeze(0)


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


class CaptionTransform(TextTokenizedTransform):

    def __init__(
        self,
        return_attn_mask: bool = True,
        shuffle: bool = True,
    ):
        self.return_attn_mask = return_attn_mask
        self.shuffle = shuffle

    def set_tokenizer(self, tokenizer: Tokenizer, max_length: int):
        self.tokenizer = deepcopy(tokenizer)
        self.tokenizer.enable_padding(length=max_length, direction="left")
        self.tokenizer.enable_truncation(max_length)

    def load(self, data: pd.Series, uid: str, shuffle: bool = True) -> str:
        # TODO:
        # 1. load feature meanings dict for corresponding dataset to uid
        # 2. shuffle keys in data series if shuffle is True
        # 3. randomly select a sentence filler from feature meanings for each key in data series
        # 4. fill each sentence with corresponding value from data series
        # 5. concatenate all sentences into one string and return it

        if "Unlabeled" in uid:
            prefix = uid[: uid.index("Unlabeled")]
        elif "Labeled" in uid:
            prefix = uid[: uid.index("Labeled")]

        sentence_templates = TEMPLATES[prefix]
        sentences = []

        for col in data.index:
            template = random.sample(sentence_templates[col], 1)
            sentences.append(template.replace("[VALUE]", str(data[col])))

        caption = (
            " ".join(random.sample(sentences, len(sentences)))
            if shuffle
            else " ".join(sentences)
        )
        return caption

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
        enc = self.tokenizer.encode(sample)
        ids = torch.as_tensor(enc.ids)

        if self.return_attn_mask:
            mask = torch.as_tensor(enc.attention_mask)
            return ids, mask

        return ids


class StructuredDataTransform(TextTokenizedTransform):

    def __init__(
        self,
        id_map: Dict[str, int],
        return_attn_mask: bool = True,
    ):
        self.id_map = id_map
        self.return_attn_mask = return_attn_mask

    def set_tokenizer(self, tokenizer: Tokenizer, max_length: int):
        self.tokenizer = deepcopy(tokenizer)
        # We don't pad or truncate structured data b/c it is the same length every time

    def load(self, data: pd.Series):
        pass  # TODO: implement this

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
        enc = self.tokenizer.encode(sample)
        ids = torch.as_tensor(enc.ids)

        return ids


class IdentityTransform(AbstractTransform):

    def load(self, path):
        raise NotImplementedError("IdentityTransform does not support loading")

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
        return sample
