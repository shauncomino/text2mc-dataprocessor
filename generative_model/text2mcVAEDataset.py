import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class text2mcVAEDataset(Dataset):
    def __init__(self, embed_builds=True, file_paths=[], tok2embedding={}, block_ignore_list=[], fixed_size=(256, 256, 256, 32)):
        self.file_paths = file_paths
        self.tok2embedding = tok2embedding
        self.block_ignore_list = set(block_ignore_list)
        self.fixed_size = fixed_size
        self.embed_builds = embed_builds
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with h5py.File(file_path, 'r') as file:
            build_folder_in_hdf5 = list(file.keys())[0]
            data = file[build_folder_in_hdf5][()]

        # Calculate the cropping or padding
        crop_sizes = [min(data.shape[dim], self.fixed_size[dim]) for dim in range(3)]
        cropped_data = data[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2]]

        # Pad the data to fixed size
        padded_data = np.zeros(self.fixed_size[:3], dtype=np.float32)  # Only 3 dimensions for non-embedded data
        offsets = [(self.fixed_size[dim] - crop_sizes[dim]) // 2 for dim in range(3)]
        slices_data = tuple(slice(offsets[dim], offsets[dim] + crop_sizes[dim]) for dim in range(3))
        padded_data[slices_data] = cropped_data

        # Prepare the mask
        mask = np.zeros_like(padded_data, dtype=np.float32)
        for i in range(padded_data.shape[0]):
            for j in range(padded_data.shape[1]):
                for k in range(padded_data.shape[2]):
                    block_token = padded_data[i, j, k]
                    if block_token not in self.block_ignore_list:
                        mask[i, j, k] = 1

        # Convert padded_data and mask to torch tensors
        padded_token_data = torch.from_numpy(padded_data).unsqueeze(0)  # Add a channel dimension for consistency
        padded_mask = torch.from_numpy(mask)

        if self.embed_builds:
            # Embed the data
            embedded_data = np.zeros((*padded_data.shape, len(next(iter(self.tok2embedding.values())))))
            for i in range(padded_data.shape[0]):
                for j in range(padded_data.shape[1]):
                    for k in range(padded_data.shape[2]):
                        block_token = int(padded_data[i, j, k])
                        if block_token in self.tok2embedding:
                            embedded_data[i, j, k] = self.tok2embedding[block_token]
            padded_embedded_data = torch.from_numpy(embedded_data).permute(3, 0, 1, 2)
            return padded_embedded_data, padded_mask
        else:
            # Return non-embedded data with mask
            return padded_token_data, padded_mask


