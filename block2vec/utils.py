import random
import pickle
import numpy as np
import torch
import torchvision
from torch.nn.functional import interpolate, grid_sample
import matplotlib.pyplot as plt


def set_seed(seed=0):
    """ Set the seed for all possible sources of randomness to allow for reproduceability. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def interpolate3D(data, shape, mode='bilinear', align_corners=False):
    d_1 = torch.linspace(-1, 1, shape[0])
    d_2 = torch.linspace(-1, 1, shape[1])
    d_3 = torch.linspace(-1, 1, shape[2])
    meshz, meshy, meshx = torch.meshgrid((d_1, d_2, d_3))
    grid = torch.stack((meshx, meshy, meshz), 3)
    grid = grid.unsqueeze(0).to(data.device)

    scaled = grid_sample(data, grid, mode=mode, align_corners=align_corners)
    return scaled


def save_pkl(obj, name, prepath='output/'):
    with open(prepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(name, prepath='output/'):
    with open(prepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def generate_non_overlapping_target_and_context_blocks(cube, radius=1):
    target_blocks = []
    context_blocks = []
    
    x_dim, y_dim, z_dim = cube.shape
    
    # Step size should be the diameter of the sub-cube (2 * radius + 1)
    step_size = 2 * radius + 1
    
    # Iterate over the cube, ensuring non-overlapping sub-cubes
    for x in range(radius, x_dim - radius, step_size):
        for y in range(radius, y_dim - radius, step_size):
            for z in range(radius, z_dim - radius, step_size):
                # The center block (target block)
                center_block = (x, y, z)
                target_blocks.append(center_block)
                
                # The surrounding context blocks
                context = []
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        for k in range(-radius, radius + 1):
                            # Skip the center block itself
                            if i == 0 and j == 0 and k == 0:
                                continue
                            context.append((x + i, y + j, z + k))
                
                context_blocks.append(context)
    
    return target_blocks, context_blocks

"""
# Example usage:
cube = np.random.randint(0, 5, size=(7, 7, 7))  # A random 7x7x7 cube with block types ranging from 0 to 4
print(cube)
radius = 1  # Example radius

target_blocks, context_blocks = generate_non_overlapping_target_and_context_blocks(cube, radius)

print("Target Blocks:")
for tb in target_blocks:
    print(tb)

print("\nContext Blocks:")
for cb in context_blocks:
    print(cb)
"""


import h5py

# Create an empty HDF5 file
with h5py.File('empty_file.h5', 'w') as file:
    # No keys (datasets or groups) are added to the file
    pass  # This block can remain empty
