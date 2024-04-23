import h5py
import mcschematic
import numpy as np
import os


def create_schematic_file(data, schem_file_path):

    schem = mcschematic.MCSchematic()
    # Iterate over the elements of the array
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                schem.setBlock((i, j, k), data[i, j, k])

    schem.save(schem_file_path, "build", mcschematic.Version.JE_1_20_1)

# # Initialize a 3D array with empty strings
# array = np.full((10, 4, 10), "minecraft:air", dtype=object)

# # Set all blocks at specific y-coordinates to the desired block
# array[1, 0, 2] = "minecraft:stone"
# array[2, 1, 4] = "minecraft:moss_block"
# array[0, 2, 3] = "minecraft:diamond_block"
# array[2, 3, 3] = "minecraft:bricks"


# create_schematic_file(array, os.getcwd())