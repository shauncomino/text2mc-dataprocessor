# Need to run this on the combined csv file, and also put this file and the slurm script in the processed_build folder.

import h5py
import mcschematic
import numpy as np
import json
import sys
import os
import glob
import pandas as pd

def convert_hdf5_file_to_numpy_array(hdf5_file: str):
    with h5py.File(hdf5_file, 'r') as file:
        # Access a specific dataset
        dataset = file[hdf5_file]
        data = dataset[:]  # Read the data into a NumPy array        
        return data
    
def count_unique_blocks(world_array):
    json_file = open("../world2vec/tok2block.json")
    data = json.load(json_file)

    unique_blocks = []

    for coordinate in np.ndindex(world_array.shape):
        block_integer = world_array[coordinate]
        block_string = data[str(block_integer)]
        if block_integer == 4000 or block_integer == 3714 or block_integer == 102:
            block_string = "minecraft:air"
        if "[" in block_string:
                blockstates = block_string.replace("[", ",").replace("]", "").split(",")
                block_string = blockstates.pop(0)
        if block_string not in unique_blocks:
            unique_blocks.append(block_string)
        if len(unique_blocks) >= 6:
            return True
    if len(unique_blocks) < 6:
        # Delete hdf5, remove it from the csv
        print("Less than 6 unique blocks found")
        return False
    
def remove_path_from_csv(csv_file, path_to_remove):
    df = pd.read_csv(csv_file)
    print("Removing path from csv file" + path_to_remove)

    parts = path_to_remove.split('_')
    index = int(parts[2].split('_')[0])

    paths = df.loc[index, 'PROCESSED_PATHS']  # Assuming file paths are separated by semicolons
    paths = paths.strip('][').replace('\'', '').replace(' ','').split(',')

    if path_to_remove in paths:
        paths.remove(path_to_remove)
        # Update the DataFrame
        df.at[index, 'PROCESSED_PATHS'] = '[\'' + '\', \''.join(paths) + '\']'

    df.to_csv(csv_file, index=False)

def main():
    hdf5_folder = sys.argv[1]
    cvs_file = sys.argv[2]

    hdf5_files = glob.glob(os.path.join(hdf5_folder, "*.h5"))
    
    for path in hdf5_files:
        integer_world_array = convert_hdf5_file_to_numpy_array(path)
        if count_unique_blocks(integer_world_array):
            print("More than 6 unique blocks found")
        else:
            remove_path_from_csv(cvs_file, path)
            os.remove(path)

            # Delete hdf5, remove it from the csv

if __name__ == "__main__":
    main()