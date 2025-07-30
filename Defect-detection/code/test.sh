#!/bin/bash
#SBATCH --account=account_name
#SBATCH --partition=aoraki_gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=4GB
#SBATCH --time=00:00:30
nvidia-smi
