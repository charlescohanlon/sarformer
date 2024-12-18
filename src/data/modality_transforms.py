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
from typing import Optional, Text, Tuple, Union, Dict
from abc import ABC, abstractmethod
import os

from PIL import Image

from einops import rearrange
from scipy.stats import iqr
import numpy as np
from transformers import T5Tokenizer
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import pandas as pd
import re

from src.data.image_augmenter import AbstractImageAugmenter
from src.utils import to_2tuple
from src.utils.data_constants import (
    NAIP_MEAN,
    NAIP_STD,
)

from src.data.templates import TEMPLATES


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
    ):
        """Unified data augmentation for FourM

        Args:
            transforms_dict (dict): Dict of transforms for each modality
            image_augmenter (AbstractImageAugmenter): Image augmenter
            resample_mode (str, optional): Resampling mode for PIL images (default: None -> uses default resampling mode for data type)
                One out of ["bilinear", "bicubic", "nearest", None].
            img_size (int): Image size for the modality
        """

        self.transforms_dict = transforms_dict
        self.image_augmenter = image_augmenter
        self.resample_mode = resample_mode

    def unified_image_augment(self, mod_dict):
        """Apply the image augmenter to all modalities where it is applicable

        Args:
            mod_dict (dict): Dict of modalities
            uid (str, optional): Unique identifier for the image augmentation

        Returns:
            dict: Transformed dict of modalities
        """

        crop_coords, flip, orig_size, target_size, rand_aug_idx = self.image_augmenter()

        mod_dict = {
            k: self.transforms_dict[k].image_augment(
                v,
                crop_coords=crop_coords,
                flip=flip,
                orig_size=orig_size,
                target_size=target_size,
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
            uid (str, optional): Unique identifier for the image augmentation

        Returns:
            dict: Transformed dict of modalities
        """
        mod_dict = {
            k: self.transforms_dict[k].preprocess(v) for k, v in mod_dict.items()
        }

        mod_dict = self.unified_image_augment(mod_dict)

        mod_dict = {
            k: self.transforms_dict[k].postprocess(v) for k, v in mod_dict.items()
        }

        return mod_dict

    def __repr__(self):
        repr = "(UnifiedDataAugmentation,\n"
        repr += ")"
        return repr


class AbstractTransform(ABC):

    @abstractmethod
    def load(self, path):
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
    def image_crop(img: Image, coords: Tuple):
        """Crop and resize an image

        :param img: Image to crop and resize
        :param coords: Coordinates of the crop (top, left, h, w)
        :return: Cropped and resized image
        """

        row, col, h, w = coords
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
        :param fill_val: Value to fill the area outside the transformed image
        :param resample_mode: mode of interpolation
        :return: Rotated image
        """

        resample_mode = get_pil_resample_mode(resample_mode)
        img = TF.rotate(img, angle=angle, interpolation=resample_mode, fill=fill_val)
        return img


class RGBTransform(ImageTransform):

    def __init__(self):
        self.rgb_mean = NAIP_MEAN
        self.rgb_std = NAIP_STD

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
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        img = self.image_crop(img, (*crop_coords, target_size, target_size))
        img = self.image_flip(img, flip)
        return img

    def postprocess(self, sample):
        sample = self.rgb_to_tensor(sample)
        sample = self.rgb_tensor_norm(sample)
        return sample


class DepthTransform(ImageTransform):

    def __init__(self, norm_ops=[]):
        self.norm_ops = norm_ops

    def depth_to_tensor(self, img):
        img = torch.Tensor(img)
        img = img.unsqueeze(0)  # 1 x H x W
        return img

    def depth_tensor_norm(self, img):
        for op_str in self.norm_ops:
            if not hasattr(DepthTransform, op_str):
                raise ValueError(f"Invalid depth norm operation: {op_str}")
            img = getattr(DepthTransform, op_str)(img)

        return img

    @staticmethod
    def depth_robust_scaling(depth, no_data_value=-999999.0):
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
    def depth_minmax_scaling(depth, no_data_value=-999999.0):
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
        depth, thresh: float = 0.0, no_data_value=-999999.0
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
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        img = self.image_crop(img, (*crop_coords, target_size, target_size))
        img = self.image_flip(img, flip)
        return img

    def postprocess(self, sample):
        sample = np.array(sample)
        sample = self.depth_to_tensor(sample)
        sample = self.depth_tensor_norm(sample)
        return sample


class TargetDistributionTransform(ImageTransform):

    def __init__(
        self,
        img_size: int,
    ):
        self.img_size = img_size

    def load(self, path):
        center_point = (self.img_size // 2, self.img_size // 2)
        target_dist = np.zeros((self.img_size, self.img_size))
        target_dist[center_point[0], center_point[1]] = 1

        return target_dist

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        img,
        crop_coords: Tuple,
        flip: bool,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx=None,
        resample_mode: str = None,
    ):
        img = self.image_crop(img, (*crop_coords, target_size, target_size))
        img = self.image_flip(img, flip)
        return img

    def postprocess(self, sample):
        sample = torch.as_tensor(sample)
        return sample.unsqueeze(0)


class MaskTransform(ImageTransform):

    def __init__(
        self,
        mask_size: int = 224,
        mask_proportion: float = 0.6,
        patch_size: int = 32,
    ):
        if mask_proportion < 0 or mask_proportion > 1:
            raise ValueError("Mask proportion must be between 0 and 1")
        if mask_size % patch_size != 0:
            raise ValueError("Mask size must be divisible by patch size")
        self.mask_size = mask_size
        self.mask_proportion = mask_proportion
        self.patch_size = patch_size

    def load(self, path):
        reduced_size = self.mask_size // self.patch_size
        mask = np.zeros((reduced_size, reduced_size))
        remove_proportion = self.mask_proportion
        num_patches = reduced_size**2
        num_patches_remove = int(remove_proportion * num_patches)

        # randomly select patches to remove
        patches_remove_idxs = np.random.choice(
            num_patches, num_patches_remove, replace=False
        )
        # set patches-to-remove to 1
        mask.flat[patches_remove_idxs] = 1

        # inflate mask to full size
        mask = mask.repeat(self.patch_size, axis=0).repeat(self.patch_size, axis=1)
        return mask

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        img,
        crop_coords: Tuple,
        flip: bool,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx=None,
        resample_mode: str = None,
    ):
        return img

    def postprocess(self, sample):
        sample = torch.as_tensor(sample, dtype=torch.bool)
        return sample.unsqueeze(0)


class CaptionTransform(AbstractTransform):

    def __init__(
        self,
        shuffle: bool = True,
        tokenizer_name: str = "t5-small",
        index_col: str = "UID",
        max_seq_len: int = 512,
    ):
        self.shuffle = shuffle
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.index_col = index_col
        self.max_seq_len = max_seq_len

    def load(self, info) -> str:
        path, uid = info
        row = pd.read_csv(path, index_col=self.index_col, dtype=str).loc[uid]
        dataset_name = uid[
            : uid.index("Unlabeled" if "Unlabeled" in uid else "Labeled")
        ]
        sentence_templates = TEMPLATES[dataset_name]
        sentences = []
        for col in row.dropna().index:
            if col in sentence_templates.keys():  # only use specified relevant columns
                template = random.choice(sentence_templates[col])
                if "[VALUE]" not in template:
                    raise ValueError(f"Template {template} does not contain [VALUE]")
                if not template.endswith("."):
                    template += "."
                sentences.append(template.replace("[VALUE]", str(row[col])))
        if self.shuffle:
            random.shuffle(sentences)
        return " ".join(sentences)

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        val,
        crop_coords: Tuple,
        flip: bool,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        return val

    def postprocess(self, sample):
        inputs = self.tokenizer.encode(
            sample,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        return inputs.squeeze(0)  # needs to be shape (seq,)


class StructuredDataTransform(AbstractTransform):

    def __init__(
        self,
        shuffle: bool = True,
        root: str = None,
        mod_dirname: str = "structured",
        tokenizer_name: str = "t5-small",
        index_col_name: str = "uid",
    ):
        self.shuffle = shuffle
        self.col_map = {  # column name -> (natural name, unit)
            "wind": ("wind strength", "meters per second"),
            "2m_dewpoint_temperature": ("2m dewpoint temperature", "celcius"),
            "2m_temperature": ("2m temperature", "celcius"),
            "surface_pressure": ("surface pressure", "pascals"),
            "total_precipitation": ("total precipitation", "meters"),
            "precipitation_type": ("precipitation type", ""),
            "snow_depth": ("snow depth", "meters of water equivalent"),
            "low_cloud_cover": ("low cloud cover", "%"),
            "medium_cloud_cover": ("medium cloud cover", "%"),
            "cloud_base_height": ("cloud base height", "meters"),
            "max_elevation": ("maximum elevation", "meters above sea level"),
            "min_elevation": ("minimum elevation", "meters above sea level"),
        }

        self.datasets = {}
        for split in os.listdir(root):
            self.datasets[split] = {}
            path = os.path.join(root, split, mod_dirname)

            if len(os.listdir(path)) != 1:
                raise ValueError(
                    "Only one class allowed in the structured data directory"
                )
            class_dir = os.listdir(path)[0]
            path = os.path.join(path, class_dir)

            if len(os.listdir(path)) == 0:
                raise ValueError(f"No dataset file found in {path}")
            if len(os.listdir(path)) != 1:
                raise ValueError("Only one dataset file allowed for structured data")
            file = os.listdir(path)[0]
            if not file.endswith(".csv"):
                raise ValueError(f"Invalid file type: {file} in {path}")

            dataset_file = os.path.join(path, file)
            self.datasets[split] = pd.read_csv(
                dataset_file, index_col=index_col_name, dtype=str
            )
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

    def load(self, path):
        split, _, _, file = path.split("/")[-4:]
        uid = file.split(".")[0]
        dataset = self.datasets[split]
        row = dataset.loc[uid]
        row = row.dropna()
        kv_strs = []
        for col_name in row.index:
            natural_name, unit = self.col_map[col_name]
            value = row[col_name]
            kv_str = f"{natural_name}: {value}"
            if unit:
                kv_str += f" {unit}"
            kv_strs.append(kv_str)

        return ",".join(random.shuffle(kv_strs) if self.shuffle else kv_strs)

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        val,
        crop_coords: Tuple,
        flip: bool,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx=None,
        resample_mode: str = None,
    ):
        return val

    def postprocess(self, sample):
        inputs = self.tokenizer.encode(sample, return_tensors="pt")
        return inputs.squeeze(0)  # needs to be shape (seq,)
