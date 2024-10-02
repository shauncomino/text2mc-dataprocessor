import os
import traceback
import h5py
from loguru import logger
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset
from block2vec_args import Block2VecArgs
import itertools

class Block2VecDataset(Dataset):
    def __init__(self, directory, tok2block: dict, context_radius: int, max_num_targets: int, build_limit: int):
        super().__init__()
        self.build_limit = build_limit
        self.tok2block = tok2block
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
        self.total_builds = len(self.files)
        self.builds_processed = 0 
        logger.info("Found {} .h5 builds.", len(self.files))
   
    def __len__(self):
        return len(self.files)

    """ Lazy-loads each build into memory. """
    def __getitem__(self, idx):
        targets, contexts = [], []
        
        try: 
            while (idx < len(self.files)) and (len(targets) < Block2VecArgs.targets_per_batch): 
                build_name = self.files[idx]
                file_path = os.path.join(self.directory, build_name)
                try: 
                    with h5py.File(file_path, "r") as file:
                        keys = file.keys()
                        if len(keys) == 0:
                            logger.info("%s failed loading: no keys." % build_name)
                        else: 
                            build_array = np.array(file[list(keys)[0]][()], dtype=np.int32)
                            logger.info("%s loaded." % build_name)
                        
                            if not has_valid_dims(build_array, self.context_radius): 
                                logger.info("%s: skipped (shape %dx%dx%d does not meet minimum dimensions required for context radius %d)." % (build_name, build_array.shape[0], build_array.shape[1], build_array.shape[2], self.context_radius))
                            else:
                                target, context = get_target_context_blocks(build_array, self.context_radius, Block2VecArgs.targets_per_build)
                                logger.info("%s: %d targets found." % (build_name, len(target)))
                                
                                for item in target[:min(len(target), Block2VecArgs.targets_per_batch - len(targets))]:
                                    targets.append(item)
                                for item in context[:min(len(context), Block2VecArgs.targets_per_batch - len(contexts))]:
                                    contexts.append(item)

                except Exception as e:
                    print(traceback.format_exc())
                    print(f"{build_name} failed loading build due to error: \"{e}\"")
                
                self.files.remove(build_name)
                self.builds_processed += 1
                idx +=1 
                logger.info("%d/%d builds processed." % (self.builds_processed, self.total_builds))
        except Exception as e: 
            print(traceback.format_exc())
            print(f"{build_name} failed loading batch due to error: \"{e}\"")
        
        
        return (targets, contexts)
    
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
    
    target_indexes = []
    context_indexes_list = [] 

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
                target_indexes.append((x, y, z))

                center_block = build[(x, y, z)]
                if (str(center_block) == "4000"): 
                    center_block = 3714

                target_blocks.append(center_block)
                
                # The surrounding context blocks
                context = []
                context_indexes = []
                for i in range(-context_radius, context_radius + 1):
                    for j in range(-context_radius, context_radius + 1):
                        for k in range(-context_radius, context_radius + 1):
                            # Skip the center block itself
                            if i == 0 and j == 0 and k == 0:
                                continue
                            context_block = build[(x + i, y + j, z + k)]
                            context_indexes.append((x + i, y + j, z + k))
                            if (str(context_block) == "4000"): 
                                context_block = 3714
                            context.append(context_block)
                context_blocks.append(context)
                context_indexes_list.append(context_indexes)
                
    return target_blocks, context_blocks