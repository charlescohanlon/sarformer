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
import json
from einops import repeat
import wandb
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Union, Tuple
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from PIL import Image
from tokenizers import Tokenizer
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    MultiScaleStructuralSimilarityIndexMeasure,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
import diffusers.schedulers as diffusers_schedulers
from src.data.image_augmenter import EmptyAugmenter
from src.vq.scheduling import DDPMScheduler, DDIMScheduler

from src.data.multimodal_dataset_folder import MultiModalDatasetFolder
import src.utils as utils
from src.data.modality_transforms import UnifiedDataTransform
from src.data.modality_info import MODALITY_INFO, MODALITY_TRANSFORMS
from src.utils import create_model
from src.utils.optim_factory import create_optimizer
from src.utils import NativeScalerWithGradNormCount as NativeScaler

import src.models.sarformer  # Needed for @register_model to work


def get_args():
    config_parser = parser = argparse.ArgumentParser(
        description="Training Config", add_help=False
    )
    parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(
        "SARFormer training script (using DDP)", add_help=True
    )
    parser.add_argument("--run_name", type=str, default="auto")

    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size per GPU (default: %(default)s). "
        "Effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--input_size",
        default=None,
        type=Optional[int],
        help="The input size for image-like modalities",
    )
    parser.add_argument(
        "--patch_size",
        default=None,
        type=Optional[int],
        help="The patch size for image-like modalities",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--save_ckpt_freq",
        default=20,
        type=int,
        help="Checkpoint saving frequency in epochs (default: %(default)s)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        help="Name of model to train (no default, must be specified)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "bf16", "fp16", "fp32"],
        help="Data type (default: %(default)s",
    )
    parser.add_argument(
        "--cond_domains",
        nargs="+",
        default=["tok_rgb@224", "tok_depth@224", "caption", "structured_data"],
        help="Input modalities (default: %(default)s)",
    )

    # Loss
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["bce", "ce", "mse"],
        default="bce",
        help="The type of loss to be computed (default: %(default)s)",
    )
    parser.add_argument(
        "--distance_type",
        type=str,
        choices=["euclidean", "manhatten", "none"],
        default="euclidean",
        help="The distance used for weighting pixel terms in the loss (default: %(default)s)",
    )

    # Weight init / fine-tune parameters
    parser.add_argument(
        "--finetune",
        default="",
        help="finetune from checkpoint (for two-stage training)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt", default="adamw", type=str, help="Optimizer (default: %(default)s)"
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        help="Optimizer epsilon (default: %(default)s)",
    )
    parser.add_argument(
        "--opt_betas",
        default=[0.9, 0.999],
        type=float,
        nargs="+",
        help="Optimizer betas (default: %(default)s)",
    )
    parser.add_argument("--compute_grad_norm", action="store_true")
    parser.add_argument(
        "--no_compute_grad_norm", action="store_false", dest="compute_grad_norm"
    )
    parser.set_defaults(compute_grad_norm=True)
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        help="Clip gradient norm (default: %(default)s)",
    )
    parser.add_argument(
        "--skip_grad",
        type=float,
        default=None,
        help="Skip update if gradient norm larger than threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="Final value of the weight decay. (Set the same value as args.weight_decay to keep weight decay value constant)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-4,
        help="Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: %(default)s)",
    )
    parser.add_argument(
        "--min_blr",
        type=float,
        default=0.0,
        help="Lower base lr bound for cyclic schedulers that hit 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--lr_and_wd_scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler type (default: %(default)s",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Epochs to warmup LR, if scheduler supports (default: %(default)s)",
    )

    # Cooldown for inverse sqrt and other "infinite" LR schedules
    parser.add_argument(
        "--cooldown_epochs",
        type=int,
        default=10,
        help="Epochs to cool down LR, if scheduler supports (default: %(default)s)",
    )

    # Text tokenizer
    parser.add_argument(
        "--text_tokenizer_path",
        default=None,
        help="Path to trained text tokenizer",
    )

    # Data
    parser.add_argument(
        "--data_csv_path",
        type=str,
        help="Path to CSV file containing structured data. (default: %(default)s)",
    )
    parser.add_argument(
        "--csv_delimiter",
        type=str,
        default="@",
        help="Delimiter for CSV file containing structured data. (default: %(default)s)",
    )
    parser.add_argument(
        "--csv_index_col",
        type=str,
        default="uid",
        help="Index column for CSV file containing structured data. (default: %(default)s)",
    )
    parser.add_argument(
        "--no_use_valid_ids", action="store_false", dest="use_valid_ids"
    )
    parser.set_defaults(use_valid_ids=True)

    # Diffusion parameters
    parser.add_argument(
        "--num_train_timesteps",
        default=1000,
        type=int,
        help="Number of diffusion steps during training (default: %(default)s)",
    )
    parser.add_argument(
        "--eval_noise_schedule",
        default="DDIMScheduler",
        type=str,
        help="Type of diffusers.schedulers noise scheduler for evaluation. (default: %(default)s)",
    )
    parser.add_argument(
        "--num_eval_timesteps",
        default=50,
        type=int,
        help="Number of diffusion steps during evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--beta_schedule",
        default="linear",
        type=str,
        help="Forward process beta schedule. linear or squaredcos_cap_v2 (default: %(default)s)",
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="Enforce SNR of beta schedule to be zero at t=T. (default: %(default)s)",
    )
    parser.add_argument(
        "--no_zero_terminal_snr", action="store_false", dest="zero_terminal_snr"
    )
    parser.set_defaults(zero_terminal_snr=True)
    parser.add_argument(
        "--cfg_scale",
        default=0.0,
        type=float,
        help="Scale of the classifier-free guidance (default: %(default)s)",
    )
    # NOTE: We don't support this in the model rn but should we?
    # parser.add_argument(
    #     "--cls_free_guidance_dropout",
    #     default=0.2,
    #     type=int,
    #     help="Condition dropout percentage during training for classifier free guidance (default: %(default)s)",
    # )
    # parser.add_argument(
    #     "--masked_cfg",
    #     action="store_true",
    #     help="Enable to perform masking on the encoded tokens. (default: %(default)s)",
    # )
    # parser.add_argument("--no_masked_cfg", action="store_false", dest="masked_cfg")
    # parser.set_defaults(masked_cfg=True)
    # parser.add_argument(
    #     "--masked_cfg_low",
    #     default=0,
    #     type=int,
    #     help="Lower bound of number of tokens to mask out (default: %(default)s)",
    # )
    # parser.add_argument(
    #     "--masked_cfg_high",
    #     default=None,
    #     type=int,
    #     help="Upper bound of number of tokens to mask out, defaults to total number of tokens minus 1 (default: %(default)s)",
    # )
    parser.add_argument(
        "--thresholding",
        default=True,
        type=bool,
        help="Whether or not to dynamically clip outputs to [-1,1]. Only affects inference time. (default: %(default)s)",
    )

    # Eval
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="/scratch/bdej/cohanlon/data/eval",
        help="Path to evaluation data (default: %(default)s)",
    )
    parser.add_argument(
        "--num_eval_metrics_samples",
        type=Optional[int],
        default=None,
        help="Number of samples to use for evaluation metrics (default: %(default)s)",
    )
    parser.add_argument(
        "--num_logged_images",
        type=int,
        default=9,
        help="Number of images to log during evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=Optional[int],
        default=None,
        help="Batch size for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--eval_freq", default=10, type=int, help="Frequency of evaluation (in epochs)"
    )
    parser.add_argument(
        "--eval_metric_freq",
        default=10,
        type=int,
        help="Frequency of metric evaluation (in epochs)",
    )
    parser.add_argument(
        "--eval_image_log_freq",
        default=10,
        type=int,
        help="Frequency of image logging (in epochs)",
    )
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument(
        "--no_dist_eval",
        action="store_false",
        dest="dist_eval",
        help="Disabling distributed evaluation",
    )
    parser.set_defaults(dist_eval=True)

    parser.add_argument(
        "--eval_only", action="store_true", help="Perform evaluation only"
    )

    # Misc.
    parser.add_argument(
        "--output_dir", default="", help="Path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training / testing"
    )

    parser.add_argument("--seed", default=0, type=int, help="Random seed ")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=False)

    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--show_user_warnings", default=False, action="store_true")

    # Wandb logging
    parser.add_argument(
        "--log_wandb",
        default=False,
        action="store_true",
        help="Log training and validation metrics to wandb",
    )
    parser.add_argument("--no_log_wandb", action="store_false", dest="log_wandb")
    parser.set_defaults(log_wandb=False)
    parser.add_argument(
        "--wandb_project", default=None, type=str, help="Project name on wandb"
    )
    parser.add_argument(
        "--wandb_entity", default=None, type=str, help="User or team name on wandb"
    )
    parser.add_argument(
        "--wandb_run_name", default="auto", type=str, help="Run name on wandb"
    )

    # Distributed training parameters
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--print_all",
        action="store_true",
        default=False,
        help="Determines whether all gpu process print or just the main one.",
    )

    # Parse config file if there is one
    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file is specified.
    args = parser.parse_args(remaining)

    # Add the config path as a final args if given
    args.config_path = args_config.config

    return args


def setup_modality_info(args):
    # Global modality info
    modality_info = MODALITY_INFO

    # Max tokens
    for mod in modality_info:
        if "type" in modality_info and modality_info[mod]["type"] == "img":
            image_size = modality_info[mod].get("input_size", args.input_size)
            patch_size = modality_info[mod].get("patch_size", args.patch_size)
            if image_size is None or patch_size is None:
                raise ValueError(
                    f"Could not find image size and patch size for modality {mod}"
                )
            num_patches = (image_size // patch_size) ** 2
            modality_info[mod]["max_tokens"] = num_patches

    return modality_info


def setup_data(args, num_tasks: int, global_rank: int, sampler_rank: int):

    # Set up shared modality info
    modality_info = setup_modality_info(args)

    # Text
    text_tokenizer = None
    if args.text_tokenizer_path:
        text_tokenizer = Tokenizer.from_file(args.text_tokenizer_path)
        max_text_tok_len = modality_info["caption"]["max_tokens"]

    # Structured data and text
    data_df = None
    if args.data_csv_path:
        data_df = pd.read_csv(
            args.data_csv_path, sep=args.csv_delimiter, index_col=args.csv_index_col
        )

    # Configure train and eval splits
    mod_with_path = None  # check arbitrary modality for ids
    for d in modality_info.values():
        if d.get("path", None) is not None:
            mod_with_path = d["path"]
            break

    train_ids, eval_ids = None, None
    if mod_with_path is None:
        # If no mods have a path we're only using mods that come from csv so provide splits by file
        if not args.eval_only:
            if args.train_ids_path is None:
                raise ValueError(
                    "Must provide train_ids_path for csv-only mods train split"
                )
            with open(args.train_ids_path, "r") as f:
                train_ids = f.read().splitlines()

        if args.eval_data_path:
            if args.eval_ids_path is None:
                raise ValueError(
                    "Must provide eval_ids_path for csv-only mods eval split"
                )
            with open(args.eval_ids_path, "r") as f:
                eval_ids = f.read().splitlines()
    else:
        # Otherwise, use intersections of csv and file (if a csv is provided) or just file-based splits
        # This assumes all file-based modalities have the same split ids
        if not args.eval_only:
            train_mod_path = os.path.join(args.data_path, mod_with_path)
            # pick arbitrary class to look at
            cls = os.listdir(train_mod_path)[0]
            train_cls_uids = [
                n.split(".")[0] for n in os.listdir(os.path.join(train_mod_path, cls))
            ]
            if args.data_csv_path:
                train_ids = list(set(train_cls_uids).intersection(set(data_df.index)))
            else:
                train_ids = train_cls_uids

        if args.eval_data_path:
            eval_mod_path = os.path.join(args.eval_data_path, mod_with_path)
            cls = os.listdir(eval_mod_path)[0]
            eval_cls_uids = [
                n.split(".")[0] for n in os.listdir(os.path.join(eval_mod_path, cls))
            ]
            if args.data_csv_path:
                eval_ids = list(set(eval_cls_uids).intersection(set(data_df.index)))
            else:
                eval_ids = eval_cls_uids

    if train_ids is not None and eval_ids is not None:
        if len(set(train_ids).intersection(set(eval_ids))) != 0:
            raise ValueError(
                "Train and eval splits have overlapping indices. Please ensure they are disjoint."
            )

    transform = UnifiedDataTransform(
        transforms_dict=MODALITY_TRANSFORMS,
        image_augmenter=EmptyAugmenter(),
    )
    if not args.eval_only:
        dataset_train = MultiModalDatasetFolder(
            root=args.data_path,
            modalities=args.all_domains,
            modality_transforms=MODALITY_TRANSFORMS,
            modality_info=modality_info,
            valid_ids=train_ids,
            transform=transform,
            tokenizer=text_tokenizer,
            max_text_tok_length=max_text_tok_len,
            data_df=data_df,
        )
        print("dataset_train size = %d" % len(dataset_train))

        num_training_steps_per_epoch = len(dataset_train) // (
            args.batch_size * args.accum_iter * num_tasks
        )

        sampler_train = DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=sampler_rank,
            shuffle=True,
            drop_last=True,
        )

        data_loader_train = DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        print("Warning: No training data loader provided.")
        num_training_steps_per_epoch = 0
        data_loader_train = None

    if args.eval_data_path:
        dataset_val = MultiModalDatasetFolder(
            root=args.eval_data_path,
            modalities=args.all_domains,
            modality_transforms=MODALITY_TRANSFORMS,
            modality_info=modality_info,
            valid_ids=eval_ids,
            transform=transform,
            tokenizer=text_tokenizer,
            max_text_tok_length=max_text_tok_len,
            data_df=data_df,
        )
        print("dataset_val size = %d" % len(dataset_val))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = SequentialSampler(dataset_val)

        data_loader_val = DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.eval_batch_size or args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        if args.num_eval_metrics_samples is not None:
            dataset_metrics = MultiModalDatasetFolder(
                root=args.eval_data_path,
                modalities=args.all_domains,
                modality_transforms=MODALITY_TRANSFORMS,
                modality_info=modality_info,
                valid_ids=eval_ids,
                transform=transform,
                tokenizer=text_tokenizer,
                max_text_tok_length=max_text_tok_len,
                data_df=data_df,
                max_samples=args.num_eval_metrics_samples,
                pre_shuffle=True,
            )
            if args.dist_eval:
                if len(dataset_metrics) % num_tasks != 0:
                    print(
                        "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                        "This will slightly alter validation results as extra duplicate entries are added to achieve "
                        "equal num of samples per-process."
                    )
                sampler_metrics = DistributedSampler(
                    dataset_metrics,
                    num_replicas=num_tasks,
                    rank=global_rank,
                    shuffle=False,
                )
            else:
                sampler_metrics = SequentialSampler(dataset_metrics)

            data_loader_metrics = DataLoader(
                dataset_metrics,
                sampler=sampler_metrics,
                batch_size=args.eval_batch_size or args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
        else:
            data_loader_metrics = data_loader_val

        if args.num_logged_images is not None:
            dataset_image_log = MultiModalDatasetFolder(
                root=args.eval_data_path,
                modalities=args.all_domains,
                modality_transforms=MODALITY_TRANSFORMS,
                modality_info=modality_info,
                valid_ids=eval_ids,
                transform=transform,
                tokenizer=text_tokenizer,
                max_text_tok_length=max_text_tok_len,
                data_df=data_df,
                max_samples=args.num_logged_images,
                pre_shuffle=True,
                return_path=True,  # needed for overlaid images
            )
            # No dist eval, we only run it on the main process
            sampler_image_log = SequentialSampler(dataset_image_log)
            data_loader_image_log = DataLoader(
                dataset_image_log,
                sampler=sampler_image_log,
                batch_size=args.num_logged_images,  # num_logged_images << batch_size
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
        else:
            data_loader_image_log = data_loader_val
    else:
        print("Warning: No evaluation data loader provided.")
        data_loader_val = None
        data_loader_metrics = None
        data_loader_image_log = None

    return (
        modality_info,
        num_training_steps_per_epoch,
        data_loader_train,
        data_loader_val,
        data_loader_metrics,
        data_loader_image_log,
    )


def get_model(args, modality_info):
    """Creates and returns model from arguments"""
    print(f"Creating model: {args.model} conditioned on {args.cond_domains}")

    encoder_embeddings = {}
    for mod in args.cond_domains:
        info = modality_info[mod]
        if info.get("encoder_embedding", None) is not None:
            if info["type"] == "img":
                image_size = info.get("input_size", args.input_size)
                patch_size = info.get("patch_size", args.patch_size)
                encoder_embeddings[mod] = info["encoder_embedding"](
                    patch_size=patch_size, image_size=image_size
                )
            else:
                encoder_embeddings[mod] = info["encoder_embedding"]()
        else:
            raise ValueError(f"Encoder embedding not found for modality {mod}")

    model = create_model(
        args.model,
        encoder_embeddings=encoder_embeddings,
        modality_info=modality_info,
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        thresholding=args.thresholding,
        zero_terminal_snr=args.zero_terminal_snr,
    )

    return model


def main(args):
    # Distributed init
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    if args.dtype in ["float16", "fp16"]:
        dtype = torch.float16
    elif args.dtype in ["bfloat16", "bf16"]:
        dtype = torch.bfloat16
    elif args.dtype in ["float32", "fp32"]:
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")

    # Distributed training variables
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    # Data
    args.all_domains = args.cond_domains + ["target_distribution"]
    (
        modality_info,
        num_training_steps_per_epoch,
        data_loader_train,
        data_loader_val,
        data_loader_metrics,
        data_loader_image_log,
    ) = setup_data(args, num_tasks, global_rank, sampler_rank)

    # Logger
    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
        log_writer.set_step(0)
    else:
        log_writer = None

    print(args)

    # Model
    model = get_model(args, modality_info)

    # Starting from pre-trained model
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        # Remove pos_emb
        checkpoint["model"] = {
            k: v for k, v in checkpoint["model"].items() if ".pos_emb" not in k
        }

        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(msg)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.to(device, non_blocking=True)
    model_without_ddp = model

    print(f"Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")

    batch_size_no_accum = args.batch_size * utils.get_world_size()
    total_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256
    args.min_lr = args.min_blr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Total (effective) batch size = %d" % total_batch_size)
    print("Accumulate grad iterations = %d" % args.accum_iter)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print(
        "Number of training examples per epoch = %d"
        % (batch_size_no_accum * num_training_steps_per_epoch)
    )

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler(enabled=dtype == torch.float16)

    # LR and WD schedules
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    main_schedule_epochs = args.epochs

    if args.lr_and_wd_scheduler == "cosine":
        lr_schedule_values = utils.cosine_scheduler(
            args.lr,
            args.min_lr,
            main_schedule_epochs,
            num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs,
        )
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay,
            args.weight_decay_end,
            main_schedule_epochs,
            num_training_steps_per_epoch,
        )
    elif "inverse_sqrt" in args.lr_and_wd_scheduler:
        try:
            timescale = int(args.lr_and_wd.split("-")[-1])
        except:
            timescale = 10_000
        lr_schedule_values = utils.inverse_sqrt_scheduler(
            args.lr,
            args.min_lr,
            main_schedule_epochs,
            num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs,
            cooldown_epochs=args.cooldown_epochs,
            timescale=timescale,
        )
        wd_schedule_values = utils.inverse_sqrt_scheduler(
            args.weight_decay,
            args.weight_decay_end,
            main_schedule_epochs,
            num_training_steps_per_epoch,
            cooldown_epochs=args.cooldown_epochs,
            timescale=timescale,
        )
    else:
        raise NotImplementedError(
            f"Scheduler {args.lr_and_wd_scheduler} not implemented."
        )

    if len(wd_schedule_values) > 0:
        print(
            "Max WD = %.7f, Min WD = %.7f"
            % (max(wd_schedule_values), min(wd_schedule_values))
        )

    # Auto-load from checkpoint (if args say to do so)
    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    # Evaluation noise scheduler
    if args.eval_noise_schedule in ["DDPMScheduler", "DDIMScheduler"]:
        eval_noise_schedule = getattr(sys.modules[__name__], args.eval_noise_schedule)(
            num_train_timesteps=args.num_eval_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type="sample",  # Only type supported atm
            thresholding=args.thresholding,
            clip_sample=False,  # Doesn't make sense for our use case
            zero_terminal_snr=args.zero_terminal_snr,
        )
    elif args.eval_noise_schedule is not None:
        eval_noise_schedule = getattr(diffusers_schedulers, args.eval_noise_schedule)(
            num_train_timesteps=args.num_eval_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type="sample",  # Only type supported atm
            thresholding=args.thresholding,
            clip_sample=False,  # Doesn't make sense for our use case
        )
    else:
        eval_noise_schedule = None

    # Eval (trained model)
    if args.eval_only:
        if data_loader_val is None:
            raise ValueError("No evaluation data loader provided for eval only mode.")
        eval_stats = launch_evals(
            launch_eval=True,
            launch_eval_metrics=True,
            launch_eval_image_log=True,
            model=model,
            device=device,
            cond_domains=args.cond_domains,
            eval_noise_schedule=eval_noise_schedule,
            num_eval_timesteps=args.num_eval_timesteps,
            eval_data_root=args.eval_data_path,
            dtype=dtype,
            data_loader_val=data_loader_val,
            data_loader_metrics=data_loader_metrics,
            data_loader_image_log=data_loader_image_log,
            num_logged_images=args.num_logged_images,
            loss_type=args.loss_type,
            distance_type=args.distance_type,
            cfg_scale=args.cfg_scale,
            log_writer=log_writer,
            compute_metrics_on_cpu=device.type == "cpu",
        )
        if log_writer is not None:
            log_writer.update(eval_stats)
        exit(0)
    elif data_loader_train is None:
        raise ValueError("No training data loader provided but in training mode.")

    # Training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            loss_type=args.loss_type,
            distance_type=args.distance_type,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            accum_iter=args.accum_iter,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            cond_domains=args.cond_domains,
            dtype=dtype,
            output_dir=args.output_dir,
            compute_grad_norm=args.compute_grad_norm,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    ckpt_name="final" if epoch + 1 == args.epochs else None,
                )

        log_stats = {
            **{k: v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        # Evaluation
        is_last_epoch = epoch + 1 == args.epochs
        launch_evaluate = (
            (data_loader_val is not None)
            and args.eval_freq > 0
            and (epoch % args.eval_freq == 0 or is_last_epoch)
        )
        launch_eval_metrics = (
            (data_loader_metrics is not None)
            and args.eval_metrics_freq > 0
            and (epoch % args.eval_metrics_freq == 0 or is_last_epoch)
        )
        launch_eval_image_log = (
            (data_loader_image_log is not None)
            and args.eval_image_log_freq > 0
            and (epoch % args.eval_image_log_freq == 0 or is_last_epoch)
        )
        eval_stats = launch_evals(
            launch_eval=launch_evaluate,
            launch_eval_metrics=launch_eval_metrics,
            launch_eval_image_log=launch_eval_image_log,
            model=model,
            device=device,
            cond_domains=args.cond_domains,
            eval_noise_schedule=eval_noise_schedule,
            num_eval_timesteps=args.num_eval_timesteps,
            dtype=dtype,
            eval_data_root=args.eval_data_path,
            data_loader_val=data_loader_val,
            data_loader_metrics=data_loader_metrics,
            data_loader_image_log=data_loader_image_log,
            num_logged_images=args.num_logged_images,
            loss_type=args.loss_type,
            distance_type=args.distance_type,
            cfg_scale=args.cfg_scale,
            log_writer=log_writer,
            compute_metrics_on_cpu=device.type == "cpu",
        )
        log_stats.update(eval_stats)

        if log_writer is not None:
            log_writer.update(log_stats)

        if args.output_dir and utils.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_type: str,
    distance_type: str,
    device: torch.device,
    epoch: int,
    loss_scaler,
    accum_iter,
    max_norm: float = None,
    max_skip_norm: float = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    cond_domains: List[str] = [],
    dtype: torch.dtype = torch.float16,
    output_dir=None,
    compute_grad_norm=True,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for step, x in enumerate(
        metric_logger.log_every(data_loader, print_freq, header=header)
    ):
        # Assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        update_grad = (step + 1) % accum_iter == 0

        if step % accum_iter == 0:
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for param_group in optimizer.param_groups:
                    if lr_schedule_values is not None:
                        param_group["lr"] = (
                            lr_schedule_values[it] * param_group["lr_scale"]
                        )
                    if (
                        wd_schedule_values is not None
                        and param_group["weight_decay"] > 0
                    ):
                        param_group["weight_decay"] = wd_schedule_values[it]

        mod_dict = {
            mod: to_device(t, device, non_blocking=True)
            for mod, t in x.items()
            if mod in cond_domains
        }

        target_distribution = x["target_distribution"].to(
            device, dtype=torch.float32, non_blocking=True
        )

        # Sample a uniformly random timestep for each image
        timesteps = torch.randint(
            0,
            unwrap_model(model).noise_scheduler.config.num_train_timesteps,
            (target_distribution.shape[0],),
        ).long()

        # Sample noise that we'll add to the images
        noise = torch.randn(target_distribution.shape).to(device, non_blocking=True)

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = (
            unwrap_model(model)
            .noise_scheduler.add_noise(target_distribution, noise, timesteps)
            .float()  # was producing float64 noisy images for some reason
        )

        # We're implicitly setting the prediction type to "sample" (i.e., x0) here
        target = target_distribution  # could try predicting the noise or velocity later

        # Only sync if we update grad (for accum_iter)
        # See https://muellerzr.github.io/blog/gradient_accumulation.html
        with nullcontext() if update_grad else model.no_sync():

            with torch.amp.autocast(device.type, dtype, enabled=dtype != torch.float32):
                logits = model(noisy_images, timesteps, mod_dict)
                loss = distance_weighted_loss(logits, target, loss_type, distance_type)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                path = os.path.join(output_dir, "debug_mod_dict.pt")
                torch.save(mod_dict, path)
                print(f"Loss is {loss_value}, stopping training", file=sys.stderr)
                print(f"Saved last mod_dict to {path}", file=sys.stderr)
                sys.exit(1)

            loss = loss / accum_iter
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                skip_grad=max_skip_norm,
                parameters=model.parameters(),
                compute_grad_norm=compute_grad_norm,
                update_grad=update_grad,
            )

            if update_grad:
                optimizer.zero_grad()

        torch.cuda.synchronize()

        # Effectively not using cache (to fit a batch size of 24)
        gc.collect()
        torch.cuda.empty_cache()

        metric_logger.update(loss=loss_value)
        min_lr = 1.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    "loss": loss_value,
                    "lr": max_lr,
                    "weight_decay": weight_decay_value,
                    "grad_norm": grad_norm,
                }
            )
            log_writer.set_step()

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {
        "[Epoch] " + k: meter.global_avg for k, meter in metric_logger.meters.items()
    }


@torch.no_grad()
def launch_evals(
    launch_eval: bool,
    launch_eval_metrics: bool,
    launch_eval_image_log: bool,
    model: Union[nn.Module, DDP],
    device: Union[torch.device, str],
    cond_domains: List[str],
    eval_noise_schedule: SchedulerMixin,
    num_eval_timesteps: int,
    eval_data_root: str,
    dtype: torch.dtype = torch.float16,
    data_loader_val: Optional[Iterable] = None,
    data_loader_metrics: Optional[Iterable] = None,
    data_loader_image_log: Optional[Iterable] = None,
    num_logged_images: int = 9,
    loss_type: str = "bce",
    distance_type: str = "euclidean",
    cfg_scale: float = 0.0,
    log_writer: Optional[utils.WandbLogger] = None,
    compute_metrics_on_cpu: bool = False,
) -> Dict[str, float]:
    """Launcher for various evaluation functions: standard evaluation,
    evaluation of image metrics, and image logging.

    Args:
        launch_eval: Whether to launch standard evaluation.
        launch_eval_metrics: Whether to launch evaluation of image metrics.
        launch_eval_image_log: Whether to launch image logging.
        model: Model to evaluate.
        device: Device to evaluate on.
        cond_domains: List of conditioning domains.
        eval_noise_schedule: Noise schedule to use for diffusion.
        num_eval_timesteps: Number of diffusion timesteps to use for evaluation.
        eval_data_root: Root directory of evaluation data.
        dtype: Data type for mixed precision inference.
        data_loader_val: Dataloader for standard evaluation.
        data_loader_metrics: Dataloader for evaluation of image metrics.
        data_loader_image_log: Dataloader for image logging.
        num_logged_images: Number of images to log.
        loss_type: Type of loss to use for evaluation.
        distance_type: Type of distance weighting to use for the loss function
        cfg_scale: Classifier-free guidance scale.
        log_writer: Optional wandb logger.
        compute_metrics_on_cpu: Whether to compute torchmetrics on CPU.

    Returns:
        Dictionary of all evaluation results.
    """
    all_eval_stats = {}

    gc.collect()
    torch.cuda.empty_cache()

    if launch_eval:
        eval_stats = eval_standard(
            model=model,
            data_loader=data_loader_val,
            device=device,
            cond_domains=cond_domains,
            distance_type=distance_type,
            loss_type=loss_type,
            dtype=dtype,
        )
        all_eval_stats.update(eval_stats)
        model.train()
        gc.collect()
        torch.cuda.empty_cache()

    # Evaluate image metrics
    if launch_eval_metrics:
        eval_metrics_results = eval_metrics(
            model=model,
            data_loader=data_loader_metrics,
            device=device,
            cond_domains=cond_domains,
            noise_schedule=eval_noise_schedule,
            num_diffusion_steps=num_eval_timesteps,
            cfg_scale=cfg_scale,
            dtype=dtype,
            compute_on_cpu=compute_metrics_on_cpu,
        )
        all_eval_stats.update(eval_metrics_results)
        model.train()
        gc.collect()
        torch.cuda.empty_cache()

    # Log images
    if launch_eval_image_log:
        eval_image_log(
            model=model,
            data_loader=data_loader_image_log,
            device=device,
            cond_domains=cond_domains,
            noise_schedule=eval_noise_schedule,
            num_diffusion_steps=num_eval_timesteps,
            eval_data_root=eval_data_root,
            cfg_scale=cfg_scale,
            dtype=dtype,
            num_logged_images=num_logged_images,
            log_writer=log_writer,
        )
        model.train()
        gc.collect()
        torch.cuda.empty_cache()

    return all_eval_stats


@torch.no_grad()
def eval_standard(
    model,
    data_loader,
    device,
    cond_domains: List[str],
    loss_type: str = "bce",
    distance_type: str = "euclidean",
    dtype: torch.dtype = torch.float16,
    header: str = "[Eval] ",
):
    # Switch to evaluation mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 10

    for x in metric_logger.log_every(data_loader, print_freq, header=header):
        mod_dict = {
            mod: to_device(t, device, non_blocking=True)
            for mod, t in x.items()
            if mod in cond_domains
        }

        target_distribution = x["target_distribution"].to(
            device, dtype=torch.float32, non_blocking=True
        )

        # Sample a uniformly random timestep for each image
        timesteps = torch.randint(
            0,
            unwrap_model(model).noise_scheduler.config.num_train_timesteps,
            (target_distribution.shape[0],),
        ).long()

        # Sample noise that we'll add to the images
        noise = torch.randn(target_distribution.shape).to(device, non_blocking=True)

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = (
            unwrap_model(model)
            .noise_scheduler.add_noise(target_distribution, noise, timesteps)
            .float()  # was producing float64 noisy images for some reason
        )

        # We're implicitly setting the prediction type to "sample" (i.e., x0) here
        target = target_distribution  # could try predicting the noise or velocity later

        with torch.amp.autocast(device.type, dtype, enabled=dtype != torch.float32):
            logits = model(noisy_images, timesteps, mod_dict)
            loss = distance_weighted_loss(logits, target, loss_type, distance_type)

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Eval averaged stats:", metric_logger)
    torch.cuda.empty_cache()

    return {header + k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_metrics(
    model: Union[nn.Module, DDP],
    data_loader: Iterable,
    device: Union[torch.device, str],
    cond_domains: List[str],
    noise_schedule: SchedulerMixin,
    num_diffusion_steps: int,
    cfg_scale: float = 0.0,
    dtype: torch.dtype = torch.float16,
    header: str = "[Eval] ",
    compute_on_cpu: bool = False,
) -> Dict[str, float]:
    """Compute (more expensive) validation image metrics (MS-SSIM, PSNR, MSE, MAE)
    using torchmetrics

    Args:
        model: Model to evaluate.
        data_loader: Validation set data loader.
        device: Device to evaluate on.
        cond_domains: List of conditioning domains.
        noise_schedule: Noise schedule to use for diffusion.
        num_diffusion_steps: Number of diffusion steps to use for eval.
        cfg_scale: Classifier-free guidance scale.
        dtype: Data type for mixed precision inference.
        header: The prefix (but includes a space after), for wandb logging.
        compute_on_cpu: Whether to compute torchmetrics on CPU.

    Returns:
        Dictionary of image metrics evaluation results.
    """
    model.eval()

    # Initialize metrics
    mse_metric = MeanSquaredError(
        squared=True, sync_on_compute=True, compute_on_cpu=compute_on_cpu
    ).to(device, non_blocking=True)
    mae_metric = MeanAbsoluteError(
        sync_on_compute=True, compute_on_cpu=compute_on_cpu
    ).to(device, non_blocking=True)
    psnr_metric = PeakSignalNoiseRatio(
        data_range=1.0,  # Input should be in [0, 1]
        reduction="elementwise_mean",
        sync_on_compute=True,
        compute_on_cpu=compute_on_cpu,
    ).to(device, non_blocking=True)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0,
        reduction="elementwise_mean",
        normalize=False,
        sync_on_compute=True,
        compute_on_cpu=compute_on_cpu,
    ).to(device, non_blocking=True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    for x in metric_logger.log_every(
        data_loader, print_freq=10, header=f"{header}Image metrics:"
    ):
        mod_dict = {
            mod: to_device(t, device, non_blocking=True)
            for mod, t in x.items()
            if mod in cond_domains
        }

        # Autoencode the images
        with torch.amp.autocast(device.type, dtype, enabled=dtype != torch.float32):
            logits = unwrap_model(model).generate(
                cond_mod_dict=mod_dict,
                scheduler=noise_schedule,
                timesteps=num_diffusion_steps,
                cfg_scale=cfg_scale,
                image_size=tuple(x["target_distribution"].shape[-2:]),
            )

        target_distribution = x["target_distribution"].to(
            device, dtype=torch.float32, non_blocking=True
        )

        gt = target_distribution
        pred = spatial_softmax(logits)

        # Compute metrics
        mse_metric.update(pred, gt)
        mae_metric.update(pred, gt)
        psnr_metric.update(pred, gt)
        ms_ssim_metric.update(pred, gt)

    # Compute and log metrics
    results = {}

    results[header + "MSE"] = mse_metric.compute().item()
    results[header + "MAE"] = mae_metric.compute().item()
    results[header + "PSNR"] = psnr_metric.compute().item()
    results[header + "MS-SSIM"] = ms_ssim_metric.compute().item()

    # Reset metrics
    mse_metric.reset()
    mae_metric.reset()
    psnr_metric.reset()
    ms_ssim_metric.reset()

    metric_logger.update(**results)

    # Gather the stats from all processes (they should already be the same since
    # we sync the torcheval metrics after every step)
    metric_logger.synchronize_between_processes()
    print(f"{header} Generation results:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_image_log(
    model: Union[nn.Module, DDP],
    data_loader: Iterable,
    device: Union[torch.device, str],
    cond_domains: List[str],
    noise_schedule: SchedulerMixin,
    num_diffusion_steps: int,
    eval_data_root: str,
    dtype: torch.dtype = torch.float16,
    num_logged_images: int = 9,
    cfg_scale: float = 0.0,
    log_writer: Optional[utils.WandbLogger] = None,
    header: str = "[Eval] ",
) -> None:
    """Log several reconstructed images to wandb.

    Args:
        model: Model to evaluate.
        data_loader: Validation set data loader.
        device: Device to evaluate on.
        cond_domains: List of conditioning domains.
        noise_schedule: Noise schedule to use for diffusion.
        num_diffusion_steps: Number of diffusion steps to use for eval.
        eval_data_root: Root directory of evaluation data.
        dtype: Data type for mixed precision inference.
        num_logged_images: Number of images to log.
        cfg_scale: Classifier-free guidance scale.
        log_writer: Wandb logger.
        header: Prefix for wandb logging.
    """

    if log_writer is None:
        print("No wandb logger provided, skipping image logging.")
        return

    model.eval()

    if utils.is_main_process():
        imgs = []
        for x in data_loader:
            mod_dict = {
                mod: to_device(t, device, non_blocking=True)
                for mod, t in x.items()
                if mod in cond_domains
            }

            # Autoencode the images
            with torch.amp.autocast(device.type, dtype, enabled=dtype != torch.float32):
                logits = unwrap_model(model).generate(
                    mod_dict,
                    scheduler=noise_schedule,
                    timesteps=num_diffusion_steps,
                    cfg_scale=cfg_scale,
                    image_size=tuple(x["target_distribution"].shape[-2:]),
                )

            target_distribution = x["target_distribution"].to(device, non_blocking=True)

            gt = target_distribution
            pred = spatial_softmax(logits)

            # Iterates over the batch dimension
            for target_dist, pred_dist, class_id, file_name in zip(
                gt, pred, x["class_id"], x["file_name"]
            ):
                rgb_img_path = os.path.join(
                    eval_data_root, "rgb", class_id, file_name + ".tif"
                )
                if not os.path.exists(rgb_img_path):
                    print(f"Image {rgb_img_path} not found, skipping")
                    continue
                imgs.append(create_overlaid_img(target_dist, pred_dist, rgb_img_path))

        # Log example images to wandb
        if len(imgs) > 0:
            log_writer.wandb_safe_log(
                {f"{header}Overlaid Images": wandb.Image(make_grid(imgs))},
                commit=False,
            )

        print(f"Logged {num_logged_images} eval images")


def spatial_softmax(x):
    # Flatten spatial dims then apply softmax then reshape back
    x = F.softmax(x.contiguous().flatten(-2), dim=-1).view_as(x)
    return x


def distance_weighted_loss(
    logits,
    target,
    loss_type="bce",
    distance_type="euclidean",
    eps=1e-6,
    reduction="mean",
):
    """
    Computes the distance-weighted loss between the predicted spatial
    distribution and the target one-hot distribution.

    Args:
        logits:  spatial distribution. Shape (B, C, H, W)
        target: Target one-hot distribution. Shape (B, C, H, W)
        loss_type: Type of loss to use. Supported types are "bce" (binary cross-entropy)
                "ce" (cross-entropy) and "mse" (mean-squared error).
        distance_type: Type of distance to use. Supported types are "euclidean", "manhattan" or "none".
        eps: Small constant to avoid division by zero.
        reduction: Reduction type for the loss. Supported types are "mean", "sum", and "none".
    """
    with torch.no_grad():
        H, W = logits.shape[-2:]

        # Row and column indice tensors
        i = repeat(torch.arange(H), "h -> 1 1 h w", w=W)
        j = repeat(torch.arange(W), "w -> 1 1 h w", h=H)

        # Index positions of the single non-zero value in the target distribution
        a = target.argmax(dim=-2).max().item()
        b = target.argmax(dim=-1).max().item()

        if distance_type == "euclidean":
            dists = torch.sqrt((i - a) ** 2 + (j - b) ** 2)  # (1, 1, H, W)
        elif distance_type == "manhattan":
            dists = torch.abs(i - a) + torch.abs(j - b)  # (1, 1, H, W)
        elif distance_type == "none":
            dists = torch.ones_like(logits)  # (B, C, H, W)
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")

    dists = (eps + dists).to(logits.device, non_blocking=True)
    pred = spatial_softmax(logits)

    if loss_type == "bce":
        loss = -dists * (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    elif loss_type == "ce":
        loss = -dists * target * torch.log(pred)
    elif loss_type == "mse":
        loss = dists * (target - pred) ** 2
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")


def unwrap_model(model: Union[nn.Module, DDP]) -> nn.Module:
    """Retrieves a model from a DDP wrapper, if necessary."""
    return model.module if hasattr(model, "module") else model


def make_grid(images: List[Image.Image]) -> Image.Image:
    """Aggregate a list of PIL images into a grid of images."""
    cols = int(np.floor(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def create_overlaid_img(
    target_dist: torch.Tensor, pred_dist: torch.Tensor, rgb_img_path: str
):
    """Creates an image with the target and predictions overlaid on the original image.
    The target distribution's single non-zero value is replaced by a black square to make it more visible.
    All distributions should sum to 1.

    Args:
        target_dist: The target distribution. Shape (1, H, W)
        pred_dist: The predicted distribution. Shape (1, H, W)
        rgb_img_path: Path to the rgb original image.
    """
    pred_alpha_mask = (255 * pred_dist.permute(1, 2, 0).float().cpu().numpy()).astype(
        np.uint8
    )

    H, W = target_dist.shape[-2:]
    # Byte array that corresponds to a yellow image
    yellow = np.concatenate(
        [255 * np.ones((H, W, 2)), np.zeros((H, W, 1))], axis=-1
    ).astype(np.uint8)

    pred_yellow_img = Image.fromarray(
        np.concatenate([yellow, pred_alpha_mask], axis=-1), mode="RGBA"
    )

    # White image with completely transparent alpha channel
    target_img_arr = np.concatenate(
        [255 * np.ones((H, W, 3)), np.zeros((H, W, 1))], axis=-1
    ).astype(np.uint8)

    # (i, j) of the single non-zero value in the target distribution
    i = target_dist.argmax(dim=-2).max().item()
    j = target_dist.argmax(dim=-1).max().item()

    margin = 1  # Pixel margin for black square
    lower_i = i - (margin if i - margin >= 0 else 0)
    upper_i = i + (1 + margin if i + margin < H else 1)
    lower_j = j - (margin if j - margin >= 0 else 0)
    upper_j = j + (1 + margin if j + margin < W else 1)

    # Create black square at the position of the 1 in the target distribution
    target_img_arr[lower_i:upper_i, lower_j:upper_j, :-1] = 0
    # Set the black square to full opacity
    target_img_arr[lower_i:upper_i, lower_j:upper_j, -1] = 255

    target_img = Image.fromarray(target_img_arr, mode="RGBA")

    im = Image.open(rgb_img_path)
    # Overlay the images
    im.paste(pred_yellow_img, (0, 0), pred_yellow_img)
    im.paste(target_img, (0, 0), target_img)

    return im


def to_device(
    data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    device: Union[torch.device, str] = None,
    dtype: torch.dtype = None,
    non_blocking: bool = True,
):
    """
    Move a tensor or a tuple of tensors to a device. Is used to handle the case when
    a modality includes an attention mask (e.g., captions)
    """
    # NOTE: for some reason the data loader returns the tuple as a list so we
    # check for not tensor instead and fix it
    if not isinstance(data, torch.Tensor):
        return (
            data[0].to(device, dtype, non_blocking),
            data[1].to(device, dtype, non_blocking),
        )
    return data.to(device, dtype, non_blocking)


if __name__ == "__main__":
    args = get_args()

    utils.setup_run_name(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
