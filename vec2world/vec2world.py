import h5py
import mcschematic
import numpy as np
import json

def create_schematic_file(data, schem_file_path):
    schem = mcschematic.MCSchematic()
    # Iterate over the elements of the array
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                schem.setBlock((i, j, k), data[i, j, k])

    schem.save(schem_file_path, "build", mcschematic.Version.JE_1_20_1)

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

integer_world_array = convert_hdf5_file_to_numpy_array("batch_1_310.h5")
string_world_array = convert_numpy_array_to_blocks(integer_world_array)
create_schematic_file(string_world_array, "")
