import h5py
import mcschematic
import numpy as np
import json
import os
import glob
import torch
from torch.utils.data import DataLoader
from text2mcVAEDataset import text2mcVAEDataset
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# Paths
hdf5_folder = '/home/shaun/projects/text2mc-dataprocessor/test_builds/reconstructions'
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

def embeddings_to_tokens(embedded_data, embedding_matrix):
    batch_size, embedding_dim, D, H, W = embedded_data.shape
    N = D * H * W
    embedded_data_flat = embedded_data.view(batch_size, embedding_dim, -1).permute(0, 2, 1).contiguous()
    embedded_data_flat = embedded_data_flat[0].numpy()  # (N, Embedding_Dim)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embedding_matrix)
    distances, indices = nbrs.kneighbors(embedded_data_flat)
    tokens = indices.flatten().reshape(D, H, W)
    return tokens

def convert_numpy_array_to_blocks(world_array, tokens_to_remove, air_token_id):
    world_array_blocks = np.empty_like(world_array, dtype=object)

    # Replace specified tokens with air_token_id
    for token_to_remove in tokens_to_remove:
        world_array[world_array == token_to_remove] = air_token_id

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

# Function to compute N most common tokens across all builds
def get_n_most_common_tokens(hdf5_files, N):
    token_counts = Counter()
    for hdf5_file in hdf5_files:
        integer_world_array = convert_hdf5_file_to_numpy_array(hdf5_file)
        tokens_flat = integer_world_array.flatten()
        token_counts.update(tokens_flat)
    most_common = token_counts.most_common(N)
    most_common_tokens = [int(token) for token, count in most_common]
    print(f"Most common tokens to remove: {most_common_tokens}")
    return most_common_tokens

# Get the N most common tokens to remove
tokens_to_remove = get_n_most_common_tokens(hdf5_files, N)

# Step 1: Convert .h5 files to original .schem files
for i, hdf5_file in enumerate(hdf5_files):
    print(f"Processing {hdf5_file} for original schematic")
    integer_world_array = convert_hdf5_file_to_numpy_array(hdf5_file)
    # Remove specified tokens by replacing them with the air token ID
    for token_to_remove in tokens_to_remove:
        integer_world_array[integer_world_array == token_to_remove] = air_token_id
    # Convert tokens to block names
    string_world_array = convert_numpy_array_to_blocks(integer_world_array, tokens_to_remove=[], air_token_id=air_token_id)
    schem_file_name = f"test{i}"
    unique_blocks = create_schematic_file(string_world_array, schem_folder_path_original, schem_file_name)
    sets_of_blocks_of_original.append(unique_blocks)