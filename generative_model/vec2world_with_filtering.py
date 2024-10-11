import h5py
import mcschematic
import numpy as np
import json
import os
import glob
import os
from torch.utils.data import DataLoader
from text2mcVAEDataset import text2mcVAEDataset
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# Paths
hdf5_folder = '/home/shaun/projects/text2mc-dataprocessor/test_builds/'
schem_folder_path_original = '/mnt/c/users/shaun/curseforge/minecraft/instances/text2mc/config/worldedit/schematics/'
schem_folder_path_reconstructed = '/mnt/c/users/shaun/curseforge/minecraft/instances/text2mc/config/worldedit/schematics/'
tok2block_file_path = '/home/shaun/projects/text2mc-dataprocessor/world2vec/tok2block.json'
block2embedding_file_path = '/home/shaun/projects/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'

sets_of_blocks_of_original = []
sets_of_blocks_of_processed = []

# Number of most common blocks to filter out
N = 3 # Adjust N as needed

# Functions
def create_schematic_file(data, schem_folder_path, schem_file_name):
    unique_blocks = set()
    schem = mcschematic.MCSchematic()
    # Iterate over the elements of the array
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                block = data[x, y, z]
                unique_blocks.add(block)
                schem.setBlock((x, y, z), block)
    schem.save(schem_folder_path, schem_file_name, mcschematic.Version.JE_1_20_1)
    return unique_blocks

def convert_hdf5_file_to_numpy_array(hdf5_file: str):
    with h5py.File(hdf5_file, 'r') as file:
        build_folder_in_hdf5 = list(file.keys())[0]
        data = file[build_folder_in_hdf5][()]
    return data

def convert_numpy_array_to_blocks(world_array, air_token_id):
    world_array_blocks = np.empty_like(world_array, dtype=object)
    for coordinate in np.ndindex(world_array.shape):
        block_integer = world_array[coordinate]
        block_string = tok2block.get(str(block_integer), 'minecraft:air')
        world_array_blocks[coordinate] = block_string

    return world_array_blocks

def trim_folder_path(folder_path):
    return folder_path.strip().lstrip('/').rstrip('/')

# Load mappings
with open(tok2block_file_path, 'r') as f:
    tok2block = json.load(f)

with open(block2embedding_file_path, 'r') as f:
    block2embedding = json.load(f)
    block2embedding = {k: np.array(v, dtype=np.float32) for k, v in block2embedding.items()}

block2tok = {v: int(k) for k, v in tok2block.items()}

# Get the air token ID
air_token_id = int(block2tok.get('minecraft:air', 0))

# Ensure output directories exist
os.makedirs(schem_folder_path_original, exist_ok=True)
os.makedirs(schem_folder_path_reconstructed, exist_ok=True)

# Get list of .h5 files
hdf5_files = glob.glob(os.path.join(hdf5_folder, "*.h5"))

# Step 1: Convert .h5 files to original .schem files
for hdf5_file in hdf5_files:
    print(f"Processing {hdf5_file} for original schematic")
    
    # Extract the prefix from the hdf5 file name (without extension)
    hdf5_file_prefix = os.path.splitext(os.path.basename(hdf5_file))[0]
    
    # Convert HDF5 to numpy array
    integer_world_array = convert_hdf5_file_to_numpy_array(hdf5_file)
    
    # Convert numpy array to blocks
    string_world_array = convert_numpy_array_to_blocks(integer_world_array, air_token_id=air_token_id)
    
    # Use the original HDF5 file prefix for the .schem file name
    schem_file_name = hdf5_file_prefix
    
    # Create the schematic file and save the unique blocks
    unique_blocks = create_schematic_file(string_world_array, schem_folder_path_original, schem_file_name)
    
    # Append the unique blocks to the list
    sets_of_blocks_of_original.append(unique_blocks)
