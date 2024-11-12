import glob
import os

# Specify the path to the folder
folder_path = "/lustre/fs1/groups/jaedo/generated_builds/"

# Find all .gif files in the folder and subfolders
gif_files = glob.glob(os.path.join(folder_path, '**', '*.gif'), recursive=True)

# Print the file paths
for file_path in gif_files:
    print(file_path)
