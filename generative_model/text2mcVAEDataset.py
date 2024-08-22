import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class text2mcVAEDataset(Dataset):
    def __init__(self, file_paths, token_to_embedding, block_ignore_list, fixed_size=(64, 64, 64, 16)):
        """
        Args:
            file_paths (list): List of paths to hdf5 files.
            token_to_embedding (dict): Mapping from block tokens to embeddings.
            block_ignore_list (list): List of block tokens to ignore (e.g., air blocks).
            fixed_size (tuple): The fixed size to which all builds will be padded.
        """
        self.file_paths = file_paths
        self.token_to_embedding = token_to_embedding
        self.block_ignore_list = set(block_ignore_list)  # Faster membership checking
        self.fixed_size = fixed_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as file:
            data = file['blocks'][:]

        # Embed and create mask
        embedded_data = np.zeros((*data.shape, len(next(iter(self.token_to_embedding.values())))))
        mask = np.zeros(data.shape, dtype=np.float32)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    block_token = data[i, j, k]
                    if block_token in self.token_to_embedding:
                        embedded_data[i, j, k] = self.token_to_embedding[block_token]
                        if block_token not in self.block_ignore_list:
                            mask[i, j, k] = 1
                    else:
                        mask[i, j, k] = 0

        # Pad the data and the mask
        padded_data = np.zeros(self.fixed_size, dtype=np.float32)
        padded_mask = np.zeros(self.fixed_size[:3], dtype=np.float32)
        offsets = [(self.fixed_size[i] - d) // 2 for i, d in enumerate(data.shape)]
        slices_data = tuple(slice(offsets[dim], offsets[dim] + data.shape[dim]) for dim in range(3))
        slices_mask = slices_data

        padded_data[slices_data] = embedded_data
        padded_mask[slices_mask] = mask

        return torch.from_numpy(padded_data), torch.from_numpy(padded_mask)
