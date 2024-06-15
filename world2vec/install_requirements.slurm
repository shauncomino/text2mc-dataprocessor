#!/bin/bash
#SBATCH --job-name=install_requirements     # Job name
#SBATCH --partition=normal                  # Partition (queue) name
#SBATCH --nodes=1                           #nodes
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH --mem=8G                            # Memory per node (GB)
#SBATCH --time=01:00:00                     # Time limit (HH:MM:SS)

# Install Python librariesules from requirements.txt
srun pip install -r requirements.txt