#!/bin/bash
#SBATCH --mem=0     # requests all available memory on node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x8      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdej-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --time=10:00:00      # hh:mm:ss for the job
#SBATCH --constraint=scratch
#SBATCH -e /u/cohanlon/sarformer/slurm_outputs/model/slurm-%j.err
#SBATCH -o /u/cohanlon/sarformer/slurm_outputs/model/slurm-%j.out
### GPU options ###
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=none     # <- or closest (ideal but takes too long to schedule)
### Name ###
#SBATCH --job-name=baseline_uniform


module reset # drop modules and explicitly load the ones needed
             # (good for job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

HOME=/u/$USER
source $HOME/.bashrc # to add micromamba to path
cd $HOME/sarformer
micromamba activate -n sarformer

echo "job is starting on `hostname`"
OMP_NUM_THREADS=128 torchrun --nproc_per_node=8 run_training_sarformer.py \
    --config cfgs/sarformer/model/misc/Baseline_uniform_regrpoint.yaml
