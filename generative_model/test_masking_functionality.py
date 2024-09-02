import torch
import numpy as np
import json
from text2mcVAEDataset import text2mcVAEDataset
from torch.utils.data import DataLoader
import h5py
import mcschematic
import numpy as np
import json


tok2block = None
block2embedding = None
block2embedding_file_path = r'block2vec/output/block2vec/embeddings.json'
tok2block_file_path = r'world2vec/tok2block.json'

def create_schematic_file(data, schem_folder_path, schem_file_name):
    schem = mcschematic.MCSchematic()
    # Iterate over the elements of the array
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                block = data[i, j, k]
                schem.setBlock((i, j, k), data[i, j, k])

    schem.save(schem_folder_path, schem_file_name, mcschematic.Version.JE_1_20_1)

def convert_hdf5_file_to_numpy_array(hdf5_file: str):
    with h5py.File(hdf5_file, 'r') as file:
        # Access a specific dataset
        dataset = file[hdf5_file]
        data = dataset[:]  # Read the data into a NumPy array        
        return data

def convert_numpy_array_to_blocks(world_array):
    json_file = open(tok2block_file_path)
    data = json.load(json_file)
    world_array_blocks = np.empty_like(world_array).astype(object)

    for coordinate in np.ndindex(world_array.shape):
        block_integer = world_array[coordinate]
        block_string = data[str(block_integer)]
        if block_integer == 4000 or block_integer == 3714 or block_integer == 0:
            block_string = "minecraft:air"
        world_array_blocks[coordinate] = block_string

    return world_array_blocks

def trim_folder_path(folder_path):
    return folder_path.strip().lstrip('/').rstrip('/')

# hdf5_file = sys.argv[1]
# schem_folder_path = "" if len(sys.argv) <= 2 else trim_folder_path(sys.argv[2])
# schem_file_name = hdf5_file.removesuffix(".h5")

# if schem_folder_path != "" and not os.path.isdir(schem_folder_path):
#     os.makedirs(schem_folder_path)

# integer_world_array = convert_hdf5_file_to_numpy_array(hdf5_file)
# string_world_array = convert_numpy_array_to_blocks(integer_world_array)
# create_schematic_file(string_world_array, schem_folder_path, schem_file_name)



# Training function call

with open(block2embedding_file_path, 'r') as j:
    block2embedding = json.loads(j.read())

with open(tok2block_file_path, 'r') as j:
    tok2block = json.loads(j.read())

# Create a new dictionary mapping tokens directly to embeddings
tok2embedding = {}

for token, block_name in tok2block.items():
    if block_name in block2embedding:
        tok2embedding[token] = block2embedding[block_name]
    else:
        print(f"Warning: Block name '{block_name}' not found in embeddings. Skipping token '{token}'.")

hdf5_filepaths = [
    r'/mnt/d/processed_builds_compressed/rar_test5_Desert+Tavern+2.h5',
    r'/mnt/d/processed_builds_compressed/rar_test6_Desert_Tavern.h5',
    r'/mnt/d/processed_builds_compressed/zip_test_0_LargeSandDunes.h5'
]

dataset = text2mcVAEDataset(file_paths=hdf5_filepaths, embed_builds=False, tok2embedding=tok2embedding, block_ignore_list=[0, 102], fixed_size=(64, 64, 64, 32))
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

for i, (build, mask) in enumerate(data_loader):
    build, mask = next(iter(data_loader))

    mask = mask.numpy().astype(int)
    masked_build = torch.clone(build).numpy().astype(int)
    build = build.numpy().astype(int)

    mask = mask.squeeze(0)
    masked_build = masked_build.squeeze(1).squeeze(0)
    build = build.squeeze(1).squeeze(0)

    masked_build *= mask

    build = convert_numpy_array_to_blocks(build)
    masked_build = convert_numpy_array_to_blocks(masked_build)

    testing_path = r'/mnt/c/Users/shaun/curseforge/minecraft/Instances/text2mc/config/worldedit/schematics'

    create_schematic_file(build, testing_path, f"not_masked{i}")
    create_schematic_file(build, testing_path, f"masked{i}")



# WHERE TO MOVE SCHEMATIC FILES TO LOAD THEM IN WORLDEDIT:
# C:\Users\shaun\curseforge\minecraft\Instances\text2mc\config\worldedit\schematics

print("yay!")