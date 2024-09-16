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
import argparse
import datetime
import os
import random
import time
from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset
from tqdm import tqdm

import fourm.utils as utils
import fourm.utils.clip as clip
from fourm.data.modality_info import MODALITY_INFO, MODALITY_TRANSFORMS_VQVAE
from fourm.vq import get_image_tokenizer
import fourm.utils.clip as clip
from fourm.data.multimodal_dataset_folder import compute_mask

FEATURE_TASKS = ["CLIP-B16", "DINOv2-B14", "DINOv2-B14-global"]
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
    ".gif",
)


def find_image_extension(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file:
                return os.path.splitext(file)[1]
    return None


class SaveVQDataset(Dataset):
    def __init__(
        self,
        root: str,
        tokens_dir: str,
        task: str,
        modality_info: dict,
        input_size: int = 224,
        mask_value: Optional[float] = None,
        task_transforms: dict = MODALITY_TRANSFORMS_VQVAE,
        resample_mode: str = "bilinear",
        corrupt_samples_log: Optional[str] = None,
        dryrun: bool = False,
    ):
        super().__init__()
        assert mask_value is not None, "Forgot to set mask value"

        self.data_root = root
        self.tokens_root = os.path.join(root, tokens_dir)
        self.input_size = input_size
        self.task = task
        self.modality_info = modality_info
        self.mask_value = mask_value
        self.task_transforms = task_transforms
        self.dryrun = dryrun

        self.loader = lambda path: Image.open(path)

        self.classes, self.class_to_idx = find_classes(os.path.join(root, task))
        if corrupt_samples_log is not None:
            task_ext = find_image_extension(os.path.join(root, task))
            self.samples = self.get_corrupt_samples(corrupt_samples_log, task_ext)
        else:
            self.samples = make_dataset(
                os.path.join(root, task), self.class_to_idx, IMG_EXTENSIONS, None
            )

    def get_corrupt_samples(self, corrupt_samples_log, task_ext):
        # Load the log file from find_corrupted_pseudolabels.py
        with open(corrupt_samples_log, "r") as f:
            corrupt_samples = f.readlines()

        # Remove the error message that was thrown and empty characters
        corrupt_samples = [sample.split(":")[-1].strip() for sample in corrupt_samples]

        # Extract the folder and file names
        corrupt_samples = [sample.split("/")[-2:] for sample in corrupt_samples]

        # Construct path
        corrupt_samples = [
            (
                os.path.join(
                    self.data_root, self.task, s[0], s[1].replace(".npy", task_ext)
                ),
                self.class_to_idx[s[0]],
            )
            for s in corrupt_samples
        ]

        return corrupt_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.loader(path)
        img = img.convert("RGB") if self.task in ["rgb", "normal"] else img

        class_id, file_id = path.split("/")[-2:]
        file_id = file_id.split(".")[0]

        tokens_path = os.path.join(self.tokens_root, class_id, f"{file_id}.npy")
        if not self.dryrun:
            os.makedirs(os.path.dirname(tokens_path), exist_ok=True)

        imgs = []
        img_mod = self.task_transforms[self.task].preprocess(img.copy())
        # NOTE: we don't augment b/c varying the rotation is problematic given the downstream
        # task is heavily dependent on orientation
        img_mod = self.task_transforms[self.task].postprocess(img_mod)

        if self.mask_value is not None:
            mask_valid = compute_mask(
                [(self.task, self.modality_info[self.task]["no_data_value"], img_mod)]
            )
            img_mod[~repeat(mask_valid, "1 h w -> c h w", c=img_mod.shape[0])] = (
                self.mask_value
            )
            # Valid regions -> 1, Masked-out regions -> -1
            mask_valid = mask_valid.float() * 2 - 1

        # Normalize image (must be done after masking)
        img_mod = getattr(self.task_transforms[self.task], f"{self.task}_tensor_norm")(
            img_mod
        )

        # mask concatenated after normalization otherwise norm function breaks
        if mask_valid is not None:
            img_mod = torch.cat([img_mod, mask_valid], dim=0)  # Concat image with mask

        imgs.append(img_mod)
        imgs = torch.stack(imgs)

        return imgs, tokens_path


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_tasks = utils.get_world_size()
    args.num_tasks = num_tasks
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    loader_task = "rgb" if args.task in FEATURE_TASKS else args.task
    dataset = SaveVQDataset(
        root=os.path.join(args.data_root, args.split),
        tokens_dir=f"{args.task}_{args.folder_suffix}",
        task=loader_task,
        modality_info=MODALITY_INFO,
        input_size=args.input_size,
        mask_value=args.mask_value,
        resample_mode=args.resample_mode,
        corrupt_samples_log=args.corrupt_samples_log,
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=False
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_dataloader,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model, _ = get_image_tokenizer(
        args.tokenizer_id, tokenizers_root=args.tokenizers_root, encoder_only=True
    )

    model.to(device)

    print(f"Starting tokenization")
    start_time = time.time()

    if global_rank == 0 and args.verbose and not args.dryrun:
        pbar = tqdm(total=len(data_loader))
    else:
        pbar = None

    for imgs_batch, tokens_paths in data_loader:

        # Filter out already saved images
        imgs_batch_filtered, tokens_paths_filtered = [], []
        for imgs, tokens_path in zip(imgs_batch, tokens_paths):
            if not os.path.exists(tokens_path) or args.corrupt_samples_log is not None:
                imgs_batch_filtered.append(imgs)
                tokens_paths_filtered.append(tokens_path)
        if len(imgs_batch_filtered) == 0:
            if pbar is not None:
                pbar.update(1)
            continue
        imgs_batch = torch.stack(imgs_batch_filtered)
        tokens_paths = tokens_paths_filtered

        # Merge batch and number of augmentation dimensions
        imgs_batch = rearrange(imgs_batch, "b n c h w -> (b n) c h w")  # n always 1

        # For efficiency, process images with batch size that might be different from loader batch size or num augmentations
        sub_batches = imgs_batch.split(args.batch_size, dim=0)

        all_tokens = []

        for sub_batch in sub_batches:
            sub_batch = sub_batch.to(device)

            with torch.no_grad():
                tokens = model.tokenize(sub_batch)
                tokens = rearrange(tokens, "b h w -> b (h w)")

            tokens = tokens.detach().cpu().numpy().astype(np.int16)
            all_tokens.append(tokens)

        # this was for when it used to iterate through crop settings
        all_tokens = np.concatenate(all_tokens)
        all_tokens = rearrange(all_tokens, "(b n) d -> b n d", n=1)

        for tokens, tokens_path in zip(all_tokens, tokens_paths):
            if args.dryrun:
                print(f"Dryrun: rank {global_rank} -> {tokens_path}")
            else:
                np.save(tokens_path, tokens)

        if pbar is not None:
            pbar.update(1)

    # torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Tokenization time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="VQ token saver")

    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default="cc12m/rgb_ViTB-UNetP4_16k_224-448",
        help="ID of tokenizer to load.",
    )
    parser.add_argument(
        "--tokenizers_root",
        type=str,
        default="./tokenizer_ckpts",
        help="Path where tokenizer checkpoints are saved.",
    )
    parser.add_argument(
        "--data_root", type=str, default="/path/to/dataset", help="Path to dataset root"
    )
    parser.add_argument("--split", type=str, default="train", help="train or val")
    parser.add_argument("--input_size", type=int, default=224, help="Image size")
    parser.add_argument("--task", type=str, default="rgb", help="Task name")
    parser.add_argument(
        "--mask_value",
        type=float,
        default=None,
        help="Optionally set masked-out regions to this value after data augs (default: %(default)s)",
    )
    parser.add_argument(
        "--resample_mode",
        type=str,
        default=None,
        help="PIL resample mode for resizing loaded images. One out of ['bilinear', 'bicubic', 'nearest', None]. (default: %(default)s)",
    )
    parser.add_argument(
        "--corrupt_samples_log",
        type=str,
        default=None,
        help="Path to log file with corrupted samples from find_corrupted_pseudolabels.py. \
              If provided, only corrupted samples will be re-tokenized.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Set to enable progress bar",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        default=False,
        help="Set to do a dry run that creates the tokens and prints the paths without saving them to disk.",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use for tokenization"
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--folder_suffix",
        type=str,
        default="dvae_BUa_224",
        help="Suffix to add to the folder under which the tokens are saved.",
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--batch_size_dataloader",
        default=None,
        type=Optional[int],
        help="Dataloader batch size (default: %(default)s). If None, uses the same as --batch_size.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (default: %(default)s)",
    )

    # Distributed parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()
    if args.batch_size_dataloader is None:
        args.batch_size_dataloader = args.batch_size
    main(args)
