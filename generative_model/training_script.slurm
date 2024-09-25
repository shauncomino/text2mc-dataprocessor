#!/bin/bash
#!/bin/bash
#SBATCH --job-name=VAE                      # Job name
#SBATCH --partition=normal                  # Partition (queue) name
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem=64GB                           # Memory per node (GB)
#SBATCH --output=VAE_results                 # Output File
#SBATCH --time=0:30:00                    # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:2

# Workaround to ensure conda initializes the shell
source /lustre/fs1/groups/jaedo/miniconda3/etc/profile.d/conda.sh

conda activate world2vec

# Commands to run Python script
# $1 is the data frame path
# $2 is the destination path

python -m torch.distributed.launch --nproc_per_node=2 train.py
