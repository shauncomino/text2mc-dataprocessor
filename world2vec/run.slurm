#!/bin/bash
#SBATCH --job-name=run_world2vecDriver      # Job name
#SBATCH --partition=normal                  # Partition (queue) name
#SBATCH --array=1-4                         # Number of array jobs
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=8G                            # Memory per node (GB)
#SBATCH --output=driver-results-%A.out   # Output File
#SBATCH --time=01:00:00                     # Time limit (HH:MM:SS)

# The anaconda3 module provides Python 3.8.3
module load anaconda/anaconda3
# Workaround to ensure conda initializes the shell
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh

# Activate the text2mc group Conda environment
conda activate /groups/jaedo/text2mc

# Commands to run Python script
# $1 is the data frame path
# $2 is the destination path
python world2vecDriver.py $1 "/lustre/fs1/groups/jaedo/unprocessed_builds" $2 $SLURM_ARRAY_TASK_ID
