#!/bin/bash

# srun --mem=0 --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --partition=gpuA100x4 --account=bdej-delta-gpu --time=00:10:00 --constraint=scratch --gpus-per-node=4 --gpu-bind=none ~/sarformer/job_scripts/tokenizers/depth/debug_depth_tokenizer.sh

module reset # drop modules and explicitly load the ones needed
             # (good for job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

HOME=/u/$USER
source $HOME/.bashrc # to add micromamba to path
cd $HOME/sarformer
micromamba activate -n sarformer

echo "job is starting on `hostname`"
OMP_NUM_THREADS=64 torchrun --nproc_per_node=4 run_training_divae.py \
	--config cfgs/sarformer/tokenizers/depth/ViTB-UNetP4_8k_224_predx0_DEBUG.yaml
