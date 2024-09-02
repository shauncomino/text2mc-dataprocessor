import os
import traceback
import h5py
from loguru import logger
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset


class Block2VecDataset(Dataset):
    def __init__(self, directory, tok2block: dict, context_radius: int, max_build_dim: int, max_num_targets: int, build_limit: int):
        super().__init__()
        self.build_limit = build_limit
        self.tok2block = tok2block
        self.max_build_dim = max_build_dim
        self.max_num_targets = max_num_targets
        self.block_frequency = dict()
        self.context_radius = context_radius
        self.directory = directory
        self.files = []
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith(".h5"):
                self.files.append(filename)
                count += 1
                if (self.build_limit != -1 and count >= build_limit): 
                    break 
                
        logger.info("Found {} .h5 builds.", len(self.files))
   
    def __len__(self):
        return len(self.files)

    """ Lazy-loads each build into memory. """
    def __getitem__(self, idx):
        build_name = self.files[idx]

        file_path = os.path.join(self.directory, self.files[idx])
        try: 
            with h5py.File(file_path, "r") as file:
                keys = file.keys()
                if len(keys) == 0:
                    logger.info("%s failed loading: no keys." % build_name)
                    return None
                else: 
                    build_array = np.array(file[list(keys)[0]][()], dtype=np.int32)

                    logger.info("%s loaded." % build_name)
                    
                    if not has_valid_dims(build_array, self.context_radius): 
                        logger.info("%s: skipped (shape %dx%dx%d does not meet minimum dimensions required for context radius %d)." % (self.files[idx], build_array.shape[0], build_array.shape[1], build_array.shape[2], self.context_radius))
                        return None
                    else:
                        #target, context = get_target_context_blocks(build_array, self.context_radius)
                        target, context = get_target_context_blocks(build_array, self.context_radius, self.max_num_targets)
                        logger.info("%s: %d targets found." % (build_name, len(target)))

                        logger.info(target.shape)

                        return (target, context, self.files[idx])
                
        except Exception as e: 
            print(traceback.format_exc())
            print(f"{build_name} failed loading due to error: \"{e}\"")
            return None

""" Check if a given build has big enough dimensions to support neighbor radius. """
def has_valid_dims(build, context_radius): 
    # Minimum build (in every dimension) supports 1 target block with the number of blocks on either side of the target at least the neighbor radius. 
    min_dimension = context_radius*2 + 1

    if (build.shape[0] < min_dimension or build.shape[1] < min_dimension or build.shape[2] < min_dimension): 
        return False
    return True

def get_target_context_blocks(build, context_radius, max_subcubes):
    target_blocks = []
    context_blocks = []
    
    x_dim, y_dim, z_dim = build.shape
    
    # Calculate the number of sub-cubes per dimension based on max_subcubes
    subcubes_per_dim = int(np.round((max_subcubes ** (1/3))))
    
    # Calculate the new step size to space out the sub-cubes
    step_size_x = max(1, (x_dim - 2 * context_radius) // subcubes_per_dim)
    step_size_y = max(1, (y_dim - 2 * context_radius) // subcubes_per_dim)
    step_size_z = max(1, (z_dim - 2 * context_radius) // subcubes_per_dim)
    
    # Iterate over the cube, spacing out the sub-cubes
    for x in range(context_radius, x_dim - context_radius, step_size_x):
        for y in range(context_radius, y_dim - context_radius, step_size_y):
            for z in range(context_radius, z_dim - context_radius, step_size_z):
                # Stop if we exceed the max number of sub-cubes
                if len(target_blocks) >= max_subcubes:
                    break

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
    
    return np.array(target_blocks), np.array(context_blocks)