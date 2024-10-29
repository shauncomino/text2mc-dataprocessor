import h5py
import mcschematic
import numpy as np
import json
import sys
import os
import glob

def create_schematic_file(data, schem_folder_path, schem_file_name):
    schem = mcschematic.MCSchematic()
    # Iterate over the elements of the array
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                schem.setBlock((i, j, k), data[i, j, k])

    schem.save(schem_folder_path, schem_file_name, mcschematic.Version.JE_1_20_1)

def convert_hdf5_file_to_numpy_array(hdf5_file):
    try:
        with h5py.File(hdf5_file, 'r') as file:
            # Print the keys to understand the structure
            print(f"Keys in the HDF5 file {hdf5_file}: {list(file.keys())}")
            
            # Assuming the dataset is the first key in the file
            dataset_name = list(file.keys())[0]
            dataset = file[dataset_name]
            numpy_array = np.array(dataset)
            return numpy_array
    except KeyError as e:
        print(f"Error accessing dataset in file {hdf5_file}: {e}")
        return None
    except OSError as e:
        print(f"Error opening file {hdf5_file}: {e}")
        return None

def convert_numpy_array_to_blocks(world_array):
    json_file = open("../world2vec/tok2block.json")
    data = json.load(json_file)
    world_array_blocks = np.empty_like(world_array).astype(object)

    for coordinate in np.ndindex(world_array.shape):
        block_integer = world_array[coordinate]
        block_string = data[str(block_integer)]
        if block_integer == 4000 or block_integer == 3714 or block_integer == 102:
            block_string = "minecraft:air"
        world_array_blocks[coordinate] = block_string

    return world_array_blocks

def trim_folder_path(folder_path):
    return folder_path.strip().lstrip('/').rstrip('/')

# Example usage
if __name__ == "__main__":
    hdf5_folder = "/mnt/c/Users/grees/OneDrive/Desktop/SD1/text2mc-dataprocessor/rendering/hdf5s"
    hdf5_files = glob.glob(os.path.join(hdf5_folder, "*.h5"))
    schem_folder_path = "/mnt/c/Users/grees/OneDrive/Desktop/SD1/text2mc-dataprocessor/vec2world/schematics"

    for path in hdf5_files:
        print("Processing " + path)
        schem_file_name = os.path.splitext(os.path.basename(path))[0]
        integer_world_array = convert_hdf5_file_to_numpy_array(path)
        if integer_world_array is not None:
            string_world_array = convert_numpy_array_to_blocks(integer_world_array)
            create_schematic_file(string_world_array, schem_folder_path, schem_file_name)

    if schem_folder_path != "" and not os.path.isdir(schem_folder_path):
        os.makedirs(schem_folder_path)
