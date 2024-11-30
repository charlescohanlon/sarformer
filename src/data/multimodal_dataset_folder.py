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
import os
import os.path
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from src.data.modality_transforms import AbstractTransform
import pandas as pd

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
    ".jpx",
    ".npy",
    ".npz",
)

UNIFIED_EXTENSIONS = IMG_EXTENSIONS + (".json", ".txt", ".json.gz")


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    valid_ids: Optional[set] = None,
    index_col: Optional[str] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    cache_path: Optional[str] = None,
) -> List[Tuple[str, int]]:
    if cache_path is not None and os.path.exists(cache_path):
        # Load cached file paths from disk if it exists
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )
    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=False)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)

                def is_valid_id(x: str) -> bool:
                    return valid_ids is None or x in valid_ids

                if fname.endswith(".csv"):
                    if index_col is None:
                        raise ValueError("index_col must be provided for csv files")
                    indices = pd.read_csv(path, index_col=index_col, dtype=str).index
                    instances.extend(
                        [
                            ((path, uid), class_index)
                            for uid in indices
                            if is_valid_id(uid)
                        ]
                    )
                elif is_valid_file(path) and is_valid_id(fname.split(".")[0]):
                    item = path, class_index
                    instances.append(item)
    if cache_path is not None:
        # Cache all file paths s.t. setting up the dataloader is instant in the future
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(instances, f)
    return instances


class MultiModalDatasetFolder(VisionDataset):
    """
    A generic multi-modal dataset loader where the samples are arranged in this way:

        root/modality_a/class_x/xxx.ext
        root/modality_a/class_y/xxy.ext
        root/modality_a/class_z/xxz.ext

        root/modality_b/class_x/xxx.ext
        root/modality_b/class_y/xxy.ext
        root/modality_b/class_z/xxz.ext

    Args:
        root (string): Root directory path.
        modalities (list): List of modalities as strings
        modality_transforms (dict): Dict of transforms for each modality
        modality_info (dict): Dict of information for each modality
        modality_paths (dict): Dict of paths to modalities
        valid_ids (list, optional): List of valid ids to load. If None, all ids are loaded.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt logs)
            both extensions and is_valid_file should not be passed.
        max_samples (int, optional): Maximum number of samples to load. If None, all samples are loaded.
        pre_shuffle (bool, optional): Whether to shuffle the sample during the init.
        return_path (bool, optional): Whether to return the paths of the samples.
    """

    def __init__(
        self,
        root: str,
        modalities: List[str],
        modality_transforms: Dict[str, AbstractTransform],
        modality_info: Dict,
        modality_paths: Optional[Dict[str, str]] = None,
        valid_ids: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        max_samples: Optional[int] = None,
        pre_shuffle: bool = False,
        return_path: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.modality_transforms = modality_transforms
        self.modality_info = modality_info
        self.return_path = return_path
        self.modalities = modalities
        self.modality_paths = {} if modality_paths is None else modality_paths
        for mod in self.modalities:
            if (
                mod not in self.modality_paths
                and modality_info[mod].get("path") is not None
            ):
                self.modality_paths[mod] = modality_info[mod]["path"]

        _, class_to_idx = self._find_classes(
            os.path.join(root, list(self.modality_paths.values())[0])
        )

        extensions = UNIFIED_EXTENSIONS if is_valid_file is None else None

        uid_set = set(valid_ids) if valid_ids is not None else None

        samples = {
            mod: (
                make_dataset(
                    directory=os.path.join(self.root, f"{self.modality_paths[mod]}"),
                    class_to_idx=class_to_idx,  # all modalities should have same class(es)
                    extensions=extensions,
                    valid_ids=uid_set,
                    index_col=modality_info[mod].get("index_col"),
                    is_valid_file=is_valid_file,
                )
            )
            for mod in self.modalities
            if self.modality_info[mod].get("path") is not None
        }

        for mod, mod_samples in samples.items():
            if len(mod_samples) == 0:
                msg = "Found 0 valid files in subfolders of: {}\n".format(
                    os.path.join(self.root, f"{self.modality_paths[mod]}")
                )
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.extensions = extensions
        self.samples = samples

        # Select random subset of dataset if so specified
        if isinstance(max_samples, int):
            total_samples = len(list(self.samples.values())[0])
            permutation = np.random.permutation(total_samples)
            for task in samples:
                self.samples[task] = [self.samples[task][i] for i in permutation][
                    :max_samples
                ]

        if pre_shuffle:
            total_samples = len(list(self.samples.values())[0])
            permutation = np.random.permutation(total_samples)
            for task in samples:
                self.samples[task] = [self.samples[task][i] for i in permutation]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def get_class_and_file(self, path: str) -> Tuple[str, str]:
        """Extracts the class and file name from a path."""
        class_id, file_name = path.split("/")[-2:]
        file_name = file_name.split(".")[0]
        return class_id, file_name

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            dict: maps modality names to sample tensors
        """
        sample_dict = {}
        for mod in self.modalities:
            info = None  # mask and target distribution do not have info
            if mod in self.samples:
                info, _ = self.samples[mod][index]
            if self.modality_info[mod]["type"] == "img":
                path_to_return = info
            sample = self.modality_transforms[mod].load(info)
            sample_dict[mod] = sample

        if self.transform is not None:
            # Applies the UnifiedDataTransform which augments the data (as well as pre and
            # post processes it)
            sample_dict = self.transform(sample_dict)

        if self.return_path:
            class_id, file_name = self.get_class_and_file(path_to_return)
            sample_dict["class_id"] = class_id
            sample_dict["file_name"] = file_name

        return sample_dict

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])
