import os

# Specify the path to the folder
folder_path = "/lustre/fs1/groups/jaedo/generated_builds/"

# Find all .gif files in the folder and subfolders
gif_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.gif'):
            gif_files.append(os.path.join(root, file))

# Print the file paths
for file_path in gif_files:
    print(file_path)