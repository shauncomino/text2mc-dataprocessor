import os
import json

def extract_unique_names(directory, output_file):
    unique_names = set()

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                unique_names.update(data)

    unique_names_list = sorted(unique_names)

    with open(output_file, 'w') as file:
        json.dump(unique_names_list, file, indent=4)

directory = "/lustre/fs1/groups/jaedo/world2vec/missing_blocks/"
output_file = "/lustre/fs1/groups/jaedo/unique_names.json"
extract_unique_names(directory, output_file)