#!/bin/bash
#SBATCH --job-name=traintesting1           # Job name
#SBATCH --output=logs/training_%j.log         # Standard output and error log (%j is the job ID)
#SBATCH --nodes=1             # It needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h100
#SBATCH --mail-type=END
#SBATCH --mail-user=nikolas.stavrou00@gmail.com
#SBATCH --ntasks-per-node=1                # It needs to match Trainer (devices=...)
#SBATCH --time=1:30:00                  # Time limit hrs:min:sec

# Load necessary modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load HDF5/1.14.0-gompi-2023a
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1

# nvidia-smi

# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Activate Rye
# export PATH="$HOME/.rye/shims:$PATH"

# Navigate to your project directory
# cd ThesisCode/

# Sync dependencies (optional)
# rye sync --no-lock
# uv sync

# Run training pytorch lightning script
# srun uv run train_precip_lightning.py
uv run train_precip_lightning.py