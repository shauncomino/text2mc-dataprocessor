import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Block2VecDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = []
        for filename in os.listdir(directory):
            if filename.endswith(".h5"):
                self.files.append(filename)
    
    def __len__(self):
        return len(self.files)
    
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
                        return (target, context)
                
        except Exception as e: 
            print(f"{self.files[idx]} failed loading due to error: \"{e}\"")
            return ([], [])
        

""" Check if a given build has big enough dimensions to support neighbor radius. """
def has_valid_dims(build, context_radius): 
    # Minimum build (in every dimension) supports 1 target block with the number of blocks on either side of the target at least the neighbor radius. 
    min_dimension = context_radius*2 + 1

    if (build.shape[0] < min_dimension or build.shape[1] < min_dimension or build.shape[2] < min_dimension): 
        return False
    return True

# Batch contains one or more builds
def custom_collate_fn(batch):
    filtered_batch = []
    for item in batch: 
        target_blocks, context_blocks = item[0], item[1]
        if not (len(target_blocks) == 0): 
            filtered_batch.append((target_blocks, context_blocks))

    return filtered_batch

def get_target_context_blocks(build, context_radius=2):
    target_coords = []
    context_coords = []
    
    x_dim, y_dim, z_dim = build.shape
    
    # Step size should be the diameter of the sub-cube (2 * radius + 1)
    step_size = 2 * context_radius + 1
    
    # Iterate over the cube, ensuring non-overlapping sub-cubes
    for x in range(context_radius, x_dim - context_radius, step_size):
        for y in range(context_radius, y_dim - context_radius, step_size):
            for z in range(context_radius, z_dim - context_radius, step_size):
                # The center block (target block)
                center_block = (x, y, z)
                target_coords.append(center_block)
                
                # The surrounding context blocks
                context = []
                for i in range(-context_radius, context_radius + 1):
                    for j in range(-context_radius, context_radius + 1):
                        for k in range(-context_radius, context_radius + 1):
                            # Skip the center block itself
                            if i == 0 and j == 0 and k == 0:
                                continue
                            context.append((x + i, y + j, z + k))
                
                context_coords.append(context)
    return target_coords, context_coords


# Remove the "if __name__ == '__main__':" on Linux system (and put it back if you get weird errors)
if __name__ == '__main__':
    directory = 'hdf5s'
    dataset = Block2VecDataset(directory)
    dataloader = DataLoader(dataset, batch_size=5, collate_fn=custom_collate_fn, num_workers=2)

    for batch in dataloader:
        
        # Each batch is a list of words from multiple files
        print(len(batch))
