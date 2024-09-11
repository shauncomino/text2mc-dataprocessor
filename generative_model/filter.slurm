#!/bin/bash
#SBATCH --job-name=run_world2vecDriver      # Job name
#SBATCH --partition=normal                  # Partition (queue) name
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=8G                            # Memory per node (GB)
#SBATCH --output=results-%A.out             # Output File
#SBATCH --time=01:00:00                     # Time limit (HH:MM:SS)

source /lustre/fs1/groups/jaedo/miniconda3/etc/profile.d/conda.sh

conda activate world2vec

# Commands to run Python script
python filter_data.py "/lustre/fs1/groups/jaedo/processed_builds/" "$CSV_FILE_PATH"