#!/bin/bash
#SBATCH --job-name=calcmetricstesting1           # Job name
#SBATCH --output=logs/calcmetrics_%j.log         # Standard output and error log (%j is the job ID)
#SBATCH --nodes=1             # It needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --ntasks-per-node=1                # It needs to match Trainer (devices=...)
#SBATCH --time=1:00:00                  # Time limit hrs:min:sec

# Load necessary modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load HDF5/1.14.0-gompi-2023a
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1

# Activate Rye
# export PATH="$HOME/.rye/shims:$PATH"

# Navigate to your project directory
# cd SmaAT-UNet-MasterBranch/

# Sync dependencies (optional)
# rye sync --no-lock

# Run test pytorch lightning script
uv run calc_metrics_test_set_ce.py