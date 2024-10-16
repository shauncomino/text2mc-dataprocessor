#!/bin/bash
#SBATCH --job-name=Axial                      # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # Number of processes (GPUs)
#SBATCH --partition=normal                  # Partition (queue) name
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH --mem=80GB                          # Memory per node (GB)
#SBATCH --output=training.out                # Output File
#SBATCH --time=36:00:00                     # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100                   # Constraint to use nodes with H100 GPUs

# Workaround to ensure conda initializes the shell
source /lustre/fs1/groups/jaedo/miniconda3/etc/profile.d/conda.sh

conda activate world2vec

# Command to run Python script
python -u train.py
