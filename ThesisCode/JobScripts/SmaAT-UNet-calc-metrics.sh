#!/bin/bash
#SBATCH --job-name=calcmetricstesting1           # Job name
#SBATCH --output=logs/calcmetrics_%j.log         # Standard output and error log (%j is the job ID)
#SBATCH --nodes=1             # It needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --mem-per-gpu=16G
#SBATCH --ntasks-per-node=1                # It needs to match Trainer (devices=...)
#SBATCH --time=3:00:00                  # Time limit hrs:min:sec

# Load necessary modules
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

# Activate Rye
export PATH="$HOME/.rye/shims:$PATH"

# Navigate to your project directory
cd SmaAT-UNet-MasterBranch/

# Sync dependencies (optional)
rye sync --no-lock

# Run test pytorch lightning script
srun rye run python calc_metrics_test_set.py