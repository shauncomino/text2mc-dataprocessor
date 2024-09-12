import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class text2mcVAEDataset(Dataset):
    def __init__(self, file_paths=[], tok2block=None, fixed_size=(256, 256, 256), max_num_targets=10, context_radius=1):
        self.file_paths = file_paths
        self.tok2block = tok2block
        self.fixed_size = fixed_size
        self.max_num_targets = max_num_targets
        self.context_radius = context_radius

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with h5py.File(file_path, 'r') as file:
            build_folder_in_hdf5 = list(file.keys())[0]
            data = file[build_folder_in_hdf5][()]
        
        # Embed and create mask
        #embedded_data = np.zeros((*data.shape, len(next(iter(self.tok2embedding.values())))))
        #mask = np.zeros(data.shape, dtype=np.float32)
        
        #max_tok = len(self.tok2embedding) - 1
        #for i in range(data.shape[0]):
            #for j in range(data.shape[1]):
                #for k in range(data.shape[2]):
                    #block_token = data[i, j, k]
                    #if str(block_token) in self.tok2embedding:
                        #embedded_data[i, j, k] = self.tok2embedding[str(block_token)]
                        #if block_token not in self.block_ignore_list:
                            #mask[i, j, k] = 1
                    #else:
                        #print(f"Warning: Block token {str(block_token)} not found in tok2block")
                        #mask[i, j, k] = 0

        # Calculate the cropping or padding
        crop_sizes = [min(data.shape[dim], self.fixed_size[dim]) for dim in range(3)]
        #embedded_data = embedded_data[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2], :]
        #mask = mask[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2]]

        # Pad the data and the mask to fixed size
        padded_data = np.zeros(self.fixed_size, dtype=np.float32)
        #padded_mask = np.zeros(self.fixed_size[:3], dtype=np.float32)

        offsets = [(self.fixed_size[dim] - crop_sizes[dim]) // 2 for dim in range(3)]
        slices_data = tuple(slice(offsets[dim], offsets[dim] + crop_sizes[dim]) for dim in range(3))

        padded_data[slices_data] = data
        #padded_mask[slices_data[:3]] = mask

        #padded_data = torch.from_numpy(padded_data).permute(3, 0, 1, 2) 
        #padded_mask = torch.from_numpy(padded_mask)

        target, context = get_target_context_blocks(padded_data, self.context_radius, self.max_num_targets)

        return padded_data, target, context

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
                #print("context subindex")
                #print(context_indexes)
                context_blocks.append(context)
                context_indexes_list.append(context_indexes)
                
    #print("context indexes:")
    #print(context_indexes)
    return target_blocks, context_blocks