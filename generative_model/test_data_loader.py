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

# Paths
hdf5_folder = '/home/shaun/projects/text2mc-dataprocessor/test_builds'
schem_folder_path_original = '/mnt/c/users/shaun/curseforge/minecraft/instances/text2mc/config/worldedit/schematics/'
schem_folder_path_reconstructed = '/mnt/c/users/shaun/curseforge/minecraft/instances/text2mc/config/worldedit/schematics/'
tok2block_file_path = '/home/shaun/projects/text2mc-dataprocessor/world2vec/tok2block.json'
block2embedding_file_path = '/home/shaun/projects/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'

sets_of_blocks_of_original = []
sets_of_blocks_of_processed = []


# Functions: create_schematic_file, convert_hdf5_file_to_numpy_array, convert_numpy_array_to_blocks, embeddings_to_tokens (as above)
def create_schematic_file(data, schem_folder_path, schem_file_name):
    unique_blocks = set()
    schem = mcschematic.MCSchematic()
    # Iterate over the elements of the array
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                block = data[i, j, k]
                unique_blocks.add(block)
                if block in tok2block.values():
                    schem.setBlock((i, j, k), data[i, j, k])
                else:
                    schem.setBlock((i, j, k), "minecraft:air")

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

def convert_numpy_array_to_blocks(world_array):
    json_file = open(tok2block_file_path)
    data = json.load(json_file)
    world_array_blocks = np.empty_like(world_array).astype(object)

    for coordinate in np.ndindex(world_array.shape):
        block_integer = world_array[coordinate]
        block_string = data[str(block_integer)]
        if block_integer == 3714:
            block_string = "minecraft:air"
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

block2tok = {v: k for k, v in tok2block.items()}

# Ensure output directories exist
os.makedirs(schem_folder_path_original, exist_ok=True)
os.makedirs(schem_folder_path_reconstructed, exist_ok=True)

# Get list of .h5 files
hdf5_files = glob.glob(os.path.join(hdf5_folder, "*.h5"))

# Step 1: Convert .h5 files to original .schem files
for i, hdf5_file in enumerate(hdf5_files):
    print(f"Processing {hdf5_file} for original schematic")
    integer_world_array = convert_hdf5_file_to_numpy_array(hdf5_file)
    string_world_array = convert_numpy_array_to_blocks(integer_world_array)
    schem_file_name = f"test{i}_original"
    unique_blocks = create_schematic_file(string_world_array, schem_folder_path_original, schem_file_name)
    sets_of_blocks_of_original.append(unique_blocks)



# Prepare the dataset
dataset = text2mcVAEDataset(
    file_paths=hdf5_files,
    block2embedding=block2embedding,
    block2tok=block2tok,
    fixed_size=(64, 64, 64)  # Adjust as needed
)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Get embedding matrix
embedding_matrix = dataset.embedding_matrix  # (Num_Tokens, Embedding_Dim)

# Step 2: Convert embedded builds back to tokens and save as reconstructed .schem files
for idx, embedded_data in enumerate(data_loader):
    embedded_data = embedded_data.to('cpu')
    print(f"Processing build {idx} for reconstruction")
    # Convert embeddings back to tokens
    tokens = embeddings_to_tokens(embedded_data, embedding_matrix)
    # Convert tokens to block names
    tokens_flat = tokens.flatten()
    block_names_flat = [tok2block.get(str(token), 'minecraft:air') for token in tokens_flat]
    block_names = np.array(block_names_flat).reshape(tokens.shape)
    # Save as .schem file
    hdf5_file = hdf5_files[idx]
    schem_file_name =  f"test{idx}_reconstructed"
    unique_blocks = create_schematic_file(block_names, schem_folder_path_reconstructed, schem_file_name)
    sets_of_blocks_of_processed.append(unique_blocks)

for i in range(0, len(sets_of_blocks_of_original)):
    print("")
    print("Original set:")
    print("")
    print(sorted(sets_of_blocks_of_original[i]))

    print("")
    print("Processed set:")
    print(sorted(sets_of_blocks_of_processed[i]))
