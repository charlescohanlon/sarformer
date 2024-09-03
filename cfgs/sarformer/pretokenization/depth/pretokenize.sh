#!/bin/bash
#SBATCH --mem=0     # requests all available memory on node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdej-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=depth_pretokenization
#SBATCH --time=24:00:00      # hh:mm:ss for the job
#SBATCH --constraint=scratch
#SBATCH -e /u/cohanlon/sarformer/slurm_outputs/pretokenization/depth/slurm-%j.err
#SBATCH -o /u/cohanlon/sarformer/slurm_outputs/pretokenization/depth/slurm-%j.out
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none     # <- or closest (ideal but takes too long to schedule)

HOME=/u/$USER
source $HOME/.bashrc # to add micromamba to path
cd $HOME/sarformer
micromamba activate -n sarformer

python save_vq_tokens.py \
    --tokenizer_id "checkpoint-final" \
    --tokenizers_root "checkpoints/tokenizers/depth/ViTB-UNetP4_8k_224_predx0" \
    --data_root "/scratch/bdej/cohanlon/untokenized" \
    --split "train" \
    --input_size 224 \
    --task "depth" \
    --mask_value -9999.0 \
    --verbose \
    --device "cuda" \
    --folder_suffix "toks" \
    --batch_size 96 \
    --num_workers 32 \
    # --dry_run # creates but doesn't save tokens