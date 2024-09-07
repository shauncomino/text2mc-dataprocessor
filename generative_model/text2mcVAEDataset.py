import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class text2mcVAEDataset(Dataset):
    def __init__(self, file_paths=[], tok2embedding={}, block_ignore_list=[], fixed_size=(256, 256, 256, 32)):
        self.file_paths = file_paths
        self.tok2embedding = tok2embedding
        self.block_ignore_list = set(block_ignore_list)
        self.fixed_size = fixed_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with h5py.File(file_path, 'r') as file:
            build_folder_in_hdf5 = list(file.keys())[0]
            data = file[build_folder_in_hdf5][()]
        
        # Embed and create mask
        embedded_data = np.zeros((*data.shape, len(next(iter(self.tok2embedding.values())))))
        mask = np.zeros(data.shape, dtype=np.float32)
        
        max_tok = len(self.tok2embedding) - 1
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    block_token = data[i, j, k]
                    if str(block_token) in self.tok2embedding:
                        embedded_data[i, j, k] = self.tok2embedding[str(block_token)]
                        if block_token not in self.block_ignore_list:
                            mask[i, j, k] = 1
                    else:
                        print(f"Warning: Block token {str(block_token)} not found in tok2block")
                        mask[i, j, k] = 0

        # Calculate the cropping or padding
        crop_sizes = [min(data.shape[dim], self.fixed_size[dim]) for dim in range(3)]
        embedded_data = embedded_data[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2], :]
        mask = mask[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2]]

        # Pad the data and the mask to fixed size
        padded_data = np.zeros(self.fixed_size, dtype=np.float32)
        padded_mask = np.zeros(self.fixed_size[:3], dtype=np.float32)

        offsets = [(self.fixed_size[dim] - crop_sizes[dim]) // 2 for dim in range(3)]
        slices_data = tuple(slice(offsets[dim], offsets[dim] + crop_sizes[dim]) for dim in range(3))

        padded_data[slices_data] = embedded_data
        padded_mask[slices_data[:3]] = mask

        padded_data = torch.from_numpy(padded_data).permute(3, 0, 1, 2) 
        padded_mask = torch.from_numpy(padded_mask)

        return padded_data, padded_mask

