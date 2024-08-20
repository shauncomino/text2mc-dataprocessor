import os
import traceback
from collections import defaultdict
from itertools import product
import h5py
from loguru import logger
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset


class Block2VecDataset(Dataset):
    def __init__(self, directory, tok2block: dict, context_radius: int):
        super().__init__()
        self.tok2block = tok2block
        self.block_frequency = dict()
        self.context_radius = context_radius
        self.directory = directory
        self.files = []
        for filename in os.listdir(directory):
            if filename.endswith(".h5"):
                self.files.append(filename)
        logger.info("Found {} .h5 builds.", len(self.files))
   
    def __len__(self):
        return len(self.files)

    """ Store discard probabilities for each token """
    def _init_discards(self):
        threshold = 0.001
        token_frequencies = list(self.block_frequency.values())
        freq = np.array(token_frequencies) / sum(token_frequencies)
        self.discards = 1.0 - (np.sqrt(freq / threshold) + 1) * (threshold / freq)

    """ Size stuff"""
    def _store_sizes(self, build): 
        x_max, y_max, z_max = build.shape

        for x in range(0, x_max):
            if build[x][0][0] == 65535:
                x_max = x
                break
        for y in range(0, y_max):
            if build[0][y][0] == 65535:
                y_max = y
                break
        for z in range(0, z_max):
            if build[0][0][z] == 65535:
                z_max = z
                break

        # All valid coordinates
        coords = np.array([(x, y, z) for x, y, z in product(range(0, x_max),
            range(0, y_max), range(0, z_max))])
        
        # Collect counts for each block 
        for coord in coords:
            block_tok = build[coord[0], coord[1], coord[2]]
            block_name = "unknown block"
            if str(block_tok) in self.tok2block:
                block_name = self.tok2block[str(block_tok)]
            else:
                block_name = block_name + " (%d)" % block_tok

            if block_name in self.block_frequency: 
                self.block_frequency[block_name] += 1
            else: 
                self.block_frequency[block_name] = 1
              
        logger.info("Found the following blocks {blocks}", blocks=dict(self.block_frequency))

    """ Lazy-loads each build into memory. """
    """ Returns tuple: (target [], context []) """
    def __getitem__(self, idx):
        context_radius = 2
        file_path = os.path.join(self.directory, self.files[idx])
        try: 
            with h5py.File(file_path, "r") as file:
                keys = file.keys()
                if len(keys) == 0:
                    print("%s failed loading: no keys." % self.files[idx])
                    return ([], [])
                else: 
                    build_array = np.array(file[list(keys)[0]][()], dtype=np.int32)
                    print("%s loaded." % self.files[idx])
                    
                    if not has_valid_dims(build_array, context_radius): 
                        print("%s: build of shape %dx%dx%d does not meet minimum dimensions required for context radius %d Skipping." % (self.files[idx], build_array.shape[0], build_array.shape[1], build_array.shape[2], context_radius))
                        return ([], [])
                    else:
                        target, context = get_target_context_blocks(build_array, context_radius)
                        print("%d targets found." % len(target))

                        self._store_sizes(build_array) 
                        return (target, context)
                
        except Exception as e: 
            print(traceback.format_exc())
            print(f"{self.files[idx]} failed loading due to error: \"{e}\"")
            return ([], [])
    
    """ Visalization of target and neighbor block context for documentation """
    def plot_coords(self, target_coord, context_coords): 
        x, y, z = zip(*context_coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
       
        ax.scatter(x, y, z, color='red')
        ax.scatter(*target_coord, color='blue')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()

# Batch contains one or more builds
def custom_collate_fn(batch):
    filtered_batch = []
    for item in batch: 
        print("item is: ", len(item[0]))
        target_blocks, context_blocks = item[0], item[1]
        if not (len(target_blocks) == 0): 
            filtered_batch.append((target_blocks, context_blocks))

    return filtered_batch

def get_target_context_blocks(build, context_radius=2):
    print("build shape is: ", build.shape)
    target_blocks = []
    context_blocks = []
    
    x_dim, y_dim, z_dim = build.shape
    
    # Step size should be the diameter of the sub-cube (2 * radius + 1)
    step_size = 2 * context_radius + 1
    
    # Iterate over the cube, ensuring non-overlapping sub-cubes
    for x in range(context_radius, x_dim - context_radius, step_size):
        for y in range(context_radius, y_dim - context_radius, step_size):
            for z in range(context_radius, z_dim - context_radius, step_size):
                # The center block (target block)
                center_block = build[(x, y, z)]
                if (str(center_block) == "4000"): 
                    center_block = 3714

                target_blocks.append(center_block)
                
                # The surrounding context blocks
                context = []
                for i in range(-context_radius, context_radius + 1):
                    for j in range(-context_radius, context_radius + 1):
                        for k in range(-context_radius, context_radius + 1):
                            # Skip the center block itself
                            if i == 0 and j == 0 and k == 0:
                                continue
                            context_block = build[(x + i, y + j, z + k)]
                            if (str(context_block) == "4000"): 
                                context_block = 3714
                            context.append(context_block)
                
                context_blocks.append(context)
    return target_blocks, context_blocks


""" Check if a given build has big enough dimensions to support neighbor radius. """
def has_valid_dims(build, context_radius): 
    # Minimum build (in every dimension) supports 1 target block with the number of blocks on either side of the target at least the neighbor radius. 
    min_dimension = context_radius*2 + 1

    if (build.shape[0] < min_dimension or build.shape[1] < min_dimension or build.shape[2] < min_dimension): 
        return False
    return True
