#!/bin/bash
#SBATCH --job-name=run_world2vec      # Job name
#SBATCH --partition=compute           # Partition (queue) name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=8G                      # Memory per node (GB)
#SBATCH --time=1-00:00:00             # Time limit (D-HH:MM:SS)

# Load any necessary modules
module load python

# Activate your virtual environment if needed
# source /path/to/your/virtualenv/bin/activate

# Install Python modules from requirements.txt
pip install -r requirements.txt

# Commands to run Python script
# $1 is the data frame path
# $2 is the starting index
# $3 is the ending index
# $4 is the batch number
python world2vecDriver.py $1 $2 $3 $4
