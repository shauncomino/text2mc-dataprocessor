import h5py
import mcschematic
import numpy as np
import json
import sys
import os

def create_schematic_file(data, schem_folder_path, schem_file_name):
    schem = mcschematic.MCSchematic()
    # Iterate over the elements of the array
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                schem.setBlock((i, j, k), data[i, j, k])

    schem.save(schem_folder_path, schem_file_name, mcschematic.Version.JE_1_20_1)

def convert_hdf5_file_to_numpy_array(hdf5_file: str):
    with h5py.File(hdf5_file, 'r') as file:
        # Access a specific dataset
        dataset = file[hdf5_file]
        data = dataset[:]  # Read the data into a NumPy array        
        return data

def convert_numpy_array_to_blocks(world_array):
    json_file = open("tok2block.json")
    data = json.load(json_file)
    world_array_blocks = np.empty_like(world_array).astype(object)

    for coordinate in np.ndindex(world_array.shape):
        block_integer = world_array[coordinate]
        block_string = data[str(block_integer)]
        world_array_blocks[coordinate] = block_string

    return world_array_blocks

hdf5_file = sys.argv[1]
schem_folder_path = "" if len(sys.argv) <= 2 else sys.argv[2]
schem_file_name = hdf5_file.removesuffix(".h5")

if schem_folder_path != "" and not os.path.isdir(schem_folder_path):
    schem_folder_path = schem_folder_path.strip().lstrip('/').rstrip('/')
    os.makedirs(schem_folder_path)

integer_world_array = convert_hdf5_file_to_numpy_array(hdf5_file)
string_world_array = convert_numpy_array_to_blocks(integer_world_array)
create_schematic_file(string_world_array, schem_folder_path, schem_file_name)
