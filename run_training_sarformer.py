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
from ast import mod
from copy import copy
import datetime
import json
from einops import repeat
from scipy import spatial
import wandb
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Union, Tuple, Callable
import gc

import matplotlib.pyplot as plt
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
from src.data import modality_transforms
from src.data.image_augmenter import CropImageAugmenter, EmptyAugmenter
from src.data.multimodal_dataset_folder import MultiModalDatasetFolder
import src.utils as utils
from src.data.modality_transforms import (
    MaskTransform,
    TargetDistributionTransform,
    UnifiedDataTransform,
)
from src.data.modality_info import MODALITY_INFO, MODALITY_TRANSFORMS
from src.utils import create_model
from src.utils.misc import destandardize
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
        "--full_img_size",
        default=673,
        type=int,
        help="The full image size for image-like modalities",
    )
    parser.add_argument(
        "--eff_img_size",
        default=None,
        type=Optional[int],
        help="The effective image size for image-like modalities, (default: full_img_size)",
    )
    parser.add_argument(
        "--target_size",
        default=224,
        type=int,
        help="The crop size for image-like modalities",
    )
    parser.add_argument(
        "--mask_proportion",
        default=0.6,
        type=float,
        help="The proportion of the image to mask for image-like modalities",
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
        default=["rgb", "depth", "caption", "structured"],
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
    parser.add_argument(
        "--objective",
        type=str,
        choices=["mae", "lpl", "lost-person location"],
        help="Objective to train the model on (default: %(default)s)",
    )

    # Augmentation
    parser.add_argument(
        "--crop_std",
        default="1e16",
        type=str,  # converted to float
        help="Standard deviation of random crop Gaussian sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--hflip",
        default=0.0,
        type=float,
        help="Probability of performing a random horizontal flip augmentation (default: %(default)s)",
    )
    parser.add_argument(
        "--vflip",
        default=0.0,
        type=float,
        help="Probability of performing a random vertical flip augmentation (default: %(default)s)",
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


def setup_data(args, num_tasks: int, global_rank: int, sampler_rank: int):

    # Set up shared modality info
    modality_info = {mod: MODALITY_INFO[mod] for mod in args.all_domains}
    modality_transforms = MODALITY_TRANSFORMS
    if "target_distribution" in args.all_domains:
        modality_transforms["target_distribution"] = TargetDistributionTransform(
            img_size=args.full_img_size
        )
    if "mask" in args.all_domains:
        modality_transforms["mask"] = MaskTransform(
            mask_size=args.target_size,
            mask_proportion=args.mask_proportion,
            patch_size=args.patch_size,
        )

    # use full_img_size if eff_img_size is not specified
    if args.eff_img_size is None:
        args.eff_img_size = args.full_img_size

    # to allow --crop_std to be scientific notation
    args.crop_std = float(args.crop_std)

    # used to set return path to get the rgb counterpart for overlaid images
    using_overlaid_imgs = args.objective == "lpl"

    transform = UnifiedDataTransform(
        transforms_dict=modality_transforms,
        image_augmenter=CropImageAugmenter(
            img_size=args.full_img_size,
            eff_img_size=args.eff_img_size,
            target_size=args.target_size,
            random_crop_std=args.crop_std,
            hflip=args.hflip,
            vflip=args.vflip,
        ),
    )
    if not args.eval_only:
        dataset_train = MultiModalDatasetFolder(
            root=args.data_path,
            modalities=args.all_domains,
            modality_transforms=modality_transforms,
            modality_info=modality_info,
            transform=transform,
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
        print("Warning: Not providing training data for eval only.")
        num_training_steps_per_epoch = 0
        data_loader_train = None

    if args.eval_data_path:
        dataset_val = MultiModalDatasetFolder(
            root=args.eval_data_path,
            modalities=args.all_domains,
            modality_transforms=modality_transforms,
            modality_info=modality_info,
            transform=transform,
        )
        print("dataset_val size = %d" % len(dataset_val))
        if len(dataset_val) == 0:
            raise ValueError("No evaluation data found.")
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
                modality_transforms=modality_transforms,
                modality_info=modality_info,
                transform=transform,
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
                modality_transforms=modality_transforms,
                modality_info=modality_info,
                transform=transform,
                max_samples=args.num_logged_images,
                pre_shuffle=True,
                return_path=using_overlaid_imgs,
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

    num_channels = sum(
        [modality_info[mod].get("num_channels", 0) for mod in args.cond_domains]
    )
    model = create_model(
        args.model,
        channels=num_channels,
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

    if args.objective == "lost-person location":  # equate acroynm to full name
        args.objective = "lpl"

    # Domains
    args.all_domains = copy(args.cond_domains)
    if args.objective == "lpl":
        args.all_domains.append("target_distribution")
        loss_fn = lambda logits, target: distance_weighted_loss(
            logits, target, args.loss_type, args.distance_type
        )
    elif args.objective == "mae":
        args.all_domains.append("mask")
        loss_fn = mae_loss
    else:
        raise ValueError(f"Objective {args.objective} not implemented.")

    # Data
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
            eval_data_root=args.eval_data_path,
            dtype=dtype,
            loss_fn=loss_fn,
            objective=args.objective,
            data_loader_val=data_loader_val,
            data_loader_metrics=data_loader_metrics,
            data_loader_image_log=data_loader_image_log,
            num_logged_images=args.num_logged_images,
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
            loss_fn=loss_fn,
            objective=args.objective,
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
            eval_data_root=args.eval_data_path,
            dtype=dtype,
            loss_fn=loss_fn,
            objective=args.objective,
            data_loader_val=data_loader_val,
            data_loader_metrics=data_loader_metrics,
            data_loader_image_log=data_loader_image_log,
            num_logged_images=args.num_logged_images,
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
    loss_fn: Callable,
    objective: str,
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

        spatial_input, seq_input = prep_inputs(x, device)

        if objective == "mae":
            # save target before masking
            target = spatial_input.detach().clone()
            # apply MAE mask (1 to remove, 0 to keep)
            mask = x["mask"].to(device, non_blocking=True).expand_as(spatial_input)
            spatial_input[mask] = 0  # remove masked values

        # Only sync if we update grad (for accum_iter)
        # See https://muellerzr.github.io/blog/gradient_accumulation.html
        with nullcontext() if update_grad else model.no_sync():
            with torch.amp.autocast(device.type, dtype, enabled=dtype != torch.float32):
                logits = model(spatial_input, seq_input)

                if objective == "lpl":
                    target = x["target_distribution"].to(device, non_blocking=True)
                    loss = loss_fn(logits, target)
                elif objective == "mae":
                    loss = loss_fn(logits, target, mask)

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    path = os.path.join(output_dir, "debug_mod_dict.pt")
                    torch.save(x, path)
                    print(f"Loss is {loss_value}, stopping training", file=sys.stderr)
                    print(f"Saved last batch to {path}", file=sys.stderr)
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
        gc.collect()

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
    eval_data_root: str,
    dtype: torch.dtype,
    loss_fn: Callable,
    objective: str,
    data_loader_val: Optional[Iterable] = None,
    data_loader_metrics: Optional[Iterable] = None,
    data_loader_image_log: Optional[Iterable] = None,
    num_logged_images: int = 9,
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
        eval_data_root: Root directory of evaluation data.
        dtype: Data type for mixed precision inference.
        loss_fn: Loss function to evaluate.
        objective: Objective to train the model on.
        data_loader_val: Dataloader for standard evaluation.
        data_loader_metrics: Dataloader for evaluation of image metrics.
        data_loader_image_log: Dataloader for image logging.
        num_logged_images: Number of images to log.
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
            dtype=dtype,
            loss_fn=loss_fn,
            objective=objective,
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
            dtype=dtype,
            compute_on_cpu=compute_metrics_on_cpu,
            objective=objective,
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
            eval_data_root=eval_data_root,
            dtype=dtype,
            num_logged_images=num_logged_images,
            log_writer=log_writer,
            objective=objective,
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
    dtype: torch.dtype,
    loss_fn: Callable,
    objective: str,
    header: str = "[Eval] ",
):
    # Switch to evaluation mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 10

    for x in metric_logger.log_every(data_loader, print_freq, header=header):
        spatial_input, seq_input = prep_inputs(x, device)

        if objective == "mae":
            # save target before masking
            target = spatial_input.detach().clone()
            # apply MAE mask (1 to remove, 0 to keep)
            mask = x["mask"].to(device, non_blocking=True).expand_as(spatial_input)
            spatial_input[mask] = 0  # remove masked values

        with torch.amp.autocast(device.type, dtype, enabled=dtype != torch.float32):
            logits = model(spatial_input, seq_input)

            if objective == "lpl":
                target = x["target_distribution"].to(device, non_blocking=True)
                loss = loss_fn(logits, target)
            elif objective == "mae":
                loss = loss_fn(logits, target, mask)

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
    dtype: torch.dtype,
    objective: str,
    header: str = "[Eval] ",
    compute_on_cpu: bool = False,
) -> Dict[str, float]:
    """Compute (more expensive) validation image metrics (MS-SSIM, PSNR, MSE, MAE)
    using torchmetrics

    Args:
        model: Model to evaluate.
        data_loader: Validation set data loader.
        device: Device to evaluate on.
        dtype: Data type for mixed precision inference.
        objective: Objective to train the model on.
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
        data_loader, print_freq=10, header=f"{header}Image Metrics:"
    ):
        spatial_input, seq_input = prep_inputs(x, device)

        if objective == "mae":
            # apply MAE mask (1 to remove, 0 to keep)
            mask = x["mask"].to(device, non_blocking=True).expand_as(spatial_input)
            spatial_input[mask] = 0  # remove masked values

        # Autoencode the images
        with torch.amp.autocast(device.type, dtype, enabled=dtype != torch.float32):
            logits = model(spatial_input, seq_input)

            if objective == "lpl":
                gt = x["target_distribution"].to(device, non_blocking=True)
                pred = spatial_softmax(logits)
            elif objective == "mae":
                gt_inputs, pred_inputs = [], []
                if "rgb" in x:
                    gt_rgb = destandardize(x["rgb"]).to(device, non_blocking=True)
                    gt_inputs.append(gt_rgb)
                    pred_rgb = destandardize(logits[:, :3])
                    pred_inputs.append(pred_rgb)
                if "depth" in x:
                    gt_depth = x["depth"].to(device, non_blocking=True)
                    gt_inputs.append(gt_depth)
                    pred_depth = logits[:, 3:] if "rgb" in x else logits
                    pred_inputs.append(pred_depth)

                gt = torch.cat(gt_inputs, dim=1)
                pred = torch.cat(pred_inputs, dim=1)

            # Compute and log metrics
            mse_metric.update(pred, gt)
            mae_metric.update(pred, gt)
            psnr_metric.update(pred, gt)
            ms_ssim_metric.update(pred, gt)
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
    print(f"{header}Metric Results:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_image_log(
    model: Union[nn.Module, DDP],
    data_loader: Iterable,
    device: Union[torch.device, str],
    eval_data_root: str,
    dtype: torch.dtype,
    objective: str,
    num_logged_images: int = 9,
    log_writer: Optional[utils.WandbLogger] = None,
    header: str = "[Eval] ",
) -> None:
    """Log several reconstructed images to wandb.

    Args:
        model: Model to evaluate.
        data_loader: Validation set data loader.
        device: Device to evaluate on.
        eval_data_root: Root directory of evaluation data.
        dtype: Data type for mixed precision inference.
        objective: Objective to train the model on.
        num_logged_images: Number of images to log.
        log_writer: Wandb logger.
        header: Prefix for wandb logging.
    """

    if log_writer is None:
        print("No wandb logger provided, skipping image logging.")
        return

    model.eval()

    if utils.is_main_process():
        if objective == "mae":
            gt_imgs = []
        pred_imgs = []
        for x in data_loader:
            if objective == "mae" and "rgb" not in x:
                print("No RGB images found, skipping image logging")
                return

            spatial_input, seq_input = prep_inputs(x, device)

            if objective == "mae":
                # save target before masking
                target = spatial_input.detach().clone()
                # apply MAE mask (1 to remove, 0 to keep)
                mask = x["mask"].to(device, non_blocking=True).expand_as(spatial_input)
                spatial_input[mask] = 0  # remove masked values

            with torch.amp.autocast(device.type, dtype, enabled=dtype != torch.float32):
                logits = model(spatial_input, seq_input)

            if objective == "lpl":
                gt = x["target_distribution"]
                pred = spatial_softmax(logits)

                # Iterates over the batch
                for target_dist, pred_dist, class_id, file_name in zip(
                    gt, pred, x["class_id"], x["file_name"]
                ):
                    rgb_img_path = os.path.join(
                        eval_data_root, "rgb", class_id, file_name + ".tif"
                    )
                    if not os.path.exists(rgb_img_path):
                        print(f"Image {rgb_img_path} not found, skipping")
                        continue
                    pred_imgs.append(
                        create_overlaid_img(target_dist, pred_dist, rgb_img_path)
                    )

            elif objective == "mae":
                gt = destandardize(target[:, :3]).float()
                pred = destandardize(
                    logits[:, :3]
                ).float()  # ensure pred and gt are same type
                kept_in_input = ~mask[:, :3]  # already checked that rgb is in x
                pred[kept_in_input] = gt[kept_in_input]
                gt_bytes = (
                    255 * gt.float().permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
                ).astype(np.uint8)
                pred_bytes = (
                    255 * pred.float().permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
                ).astype(np.uint8)
                gt_imgs.extend([Image.fromarray(img) for img in gt_bytes])
                pred_imgs.extend([Image.fromarray(img) for img in pred_bytes])

        # Log example images to wandb
        if len(gt_imgs) > 0:
            log_writer.wandb_safe_log(
                {f"{header}Ground Truth Images": wandb.Image(make_grid(gt_imgs))},
                commit=False,
            )

        if len(pred_imgs) > 0:
            log_writer.wandb_safe_log(
                {f"{header}Predicted Images": wandb.Image(make_grid(pred_imgs))},
                commit=False,
            )

        print(f"Logged {num_logged_images} eval images")


def prep_inputs(
    x: Dict[str, torch.Tensor], device: Union[str, torch.device]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Allows control over order of modalities in input
    spatial_mods = []
    if "rgb" in x:
        spatial_mods.append(x["rgb"].to(device, non_blocking=True))
    if "depth" in x:
        spatial_mods.append(x["depth"].to(device, non_blocking=True))
    # Concatenate along channel dimension
    spatial_input = torch.cat(spatial_mods, dim=1)

    seq_mods = []
    if "structured" in x:
        seq_mods.append(x["structured"].to(device, non_blocking=True))
    if "caption" in x:
        seq_mods.append(x["caption"].to(device, non_blocking=True))
    # Concatenate along the sequence dimension
    seq_input = torch.cat(seq_mods, dim=1)

    return spatial_input, seq_input


def spatial_softmax(x):
    """Apply softmax to spatial dimensions of the input tensor.

    Args:
        x: Input image tensor. Shape (B, C, H, W)
    """
    # Flatten spatial dims, apply softmax, restore shape
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
        loss = dists * F.mse_loss(pred, target, reduction="none")
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


def mae_loss(logits, target, removed_mask):
    # Only consider the masked values (those removed from the input)
    return F.mse_loss(logits[removed_mask], target[removed_mask])


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
    H, W = target_dist.shape[-2:]

    # (i, j) of the single non-zero value in the target distribution
    i = target_dist.argmax(dim=-2).max().item()
    j = target_dist.argmax(dim=-1).max().item()

    margin = 1  # Pixel margin for black square
    lower_i = i - (margin if i - margin >= 0 else 0)
    upper_i = i + (1 + margin if i + margin < H else 1)
    lower_j = j - (margin if j - margin >= 0 else 0)
    upper_j = j + (1 + margin if j + margin < W else 1)

    cmap = plt.get_cmap("plasma")
    pred = cmap(
        pred_dist.float().permute(1, 2, 0).squeeze(-1).clamp(0, 1).cpu().numpy()
    )
    pred[:, :, -1] = 0.3  # Set alpha channel to 0.3
    pred_heatmap = Image.fromarray((255 * pred).astype(np.uint8))

    # White image with completely transparent alpha channel
    target_img_arr = np.concatenate(
        [255 * np.ones((H, W, 3)), np.zeros((H, W, 1))], axis=-1
    ).astype(np.uint8)
    # Create black square at the position of the 1 in the target distribution
    target_img_arr[lower_i:upper_i, lower_j:upper_j, :-1] = 0
    # Set the black square to full opacity
    target_img_arr[lower_i:upper_i, lower_j:upper_j, -1] = 255
    target_img = Image.fromarray(target_img_arr, mode="RGBA")

    im = Image.open(rgb_img_path)
    # Overlay the images
    im.paste(pred_heatmap, (0, 0), pred_heatmap)
    im.paste(target_img, (0, 0), target_img)

    return im


if __name__ == "__main__":
    args = get_args()

    utils.setup_run_name(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
