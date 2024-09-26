#!/bin/bash
#SBATCH --job-name=VAE                      # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # Number of processes (GPUs)
#SBATCH --partition=normal                  # Partition (queue) name
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH --mem=64GB                          # Memory per node (GB)
#SBATCH --output=VAE_results                # Output File
#SBATCH --time=0:30:00                      # Time limit (HH:MM:SS)
#SBATCH --gpus-per-node=h100:1

# Workaround to ensure conda initializes the shell
source /lustre/fs1/groups/jaedo/miniconda3/etc/profile.d/conda.sh

conda activate world2vec

# Command to run Python script
python train.py
