#!/bin/bash
#!/bin/bash
#SBATCH --job-name=run_world2vecDriver      # Job name
#SBATCH --partition=normal                  # Partition (queue) name
#SBATCH --array=1-100                         # Number of array jobs
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=20G                            # Memory per node (GB)
#SBATCH --output=results-%A.out   # Output File
#SBATCH --time=72:00:00                     # Time limit (HH:MM:SS)

# Workaround to ensure conda initializes the shell
source /lustre/fs1/groups/jaedo/miniconda3/etc/profile.d/conda.sh

conda activate world2vec

export JAVA_HOME=/groups/jaedo/jdk-22.0.1
export PATH=$JAVA_HOME/bin:$PATH
export RAR_EXECUTABLE=/lustre/fs1/groups/jaedo/unrar

# Commands to run Python script
# $1 is the data frame path
# $2 is the destination path

python world2vecDriver.py "/lustre/fs1/groups/jaedo/projects_df_processed.csv" "/lustre/fs1/groups/jaedo/unprocessed_builds/builds" "/lustre/fs1/groups/jaedo/processed_builds" $SLURM_ARRAY_TASK_ID
