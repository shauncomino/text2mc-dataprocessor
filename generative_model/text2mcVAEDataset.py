# text2mcVAEDataset.py

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class text2mcVAEDataset(Dataset):
    def __init__(self, file_paths=[], block2tok={}, block_ignore_list=[], fixed_size=(64, 64, 64), augment=False):
        self.file_paths = file_paths
        self.block2tok = block2tok
        self.block_ignore_set = set(block_ignore_list)
        self.fixed_size = fixed_size  # (Depth, Height, Width)
        self.augment = augment  # Flag to enable data augmentation

        # Prepare token mappings
        self.prepare_token_mappings()

    def prepare_token_mappings(self):
        # Collect all tokens
        tokens = list(self.block2tok.values())
        tokens = [int(token) for token in tokens]

        # Ensure tokens are non-negative integers
        if min(tokens) < 0:
            raise ValueError("All block tokens must be non-negative integers.")

        self.token_set = set(tokens)

        # Find the maximum token value to determine the size of the lookup arrays
        self.max_token = max(tokens + [3714])  # Include unknown block token

        # Get the air token
        air_block_name = 'minecraft:air'
        air_token_str = self.block2tok.get(air_block_name)
        if air_token_str is None:
            raise ValueError('minecraft:air block not found in block2tok mapping')
        self.air_token = int(air_token_str)

        # Build the lookup array mapping tokens to indices
        self.lookup_array = np.full((self.max_token + 1,), self.air_token, dtype=np.int32)
        for block_name, token_str in self.block2tok.items():
            token = int(token_str)
            self.lookup_array[token] = token  # Map token to itself

        # Map unknown block token to 3714
        self.lookup_array[3714] = 3714

        # For tokens in block_ignore_set, map them to air_token
        for block_name in self.block_ignore_set:
            token_str = self.block2tok.get(block_name)
            if token_str is not None:
                token = int(token_str)
                if token <= self.max_token:
                    self.lookup_array[token] = self.air_token

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with h5py.File(file_path, 'r') as file:
            build_folder_in_hdf5 = list(file.keys())[0]
            data = file[build_folder_in_hdf5][()]

        # Convert data tokens to integers
        data_tokens = data.astype(np.int32)

        # Map tokens outside the valid range [0, max_token] to the unknown block token (3714)
        data_tokens_mapped = np.where(
            (data_tokens >= 0) & (data_tokens <= self.max_token),
            data_tokens,
            3714  # Assign unknown block token
        )

        # Map tokens to indices using the lookup array
        data_tokens_mapped = self.lookup_array[data_tokens_mapped]

        if self.augment:
            # === Data Augmentation ===

            # Random rotation around Y axis (vertical axis)
            k = np.random.choice([0, 1, 2, 3])  # Number of 90-degree rotations
            # Rotate in the (Depth, Width) plane
            data_tokens_mapped = np.rot90(data_tokens_mapped, k=k, axes=(0, 2))

            # Define maximum shifts as 1/4 of the fixed size
            max_shift_depth = self.fixed_size[0] // 4
            max_shift_height = self.fixed_size[1] // 4
            max_shift_width = self.fixed_size[2] // 4

            # Calculate padded size
            padded_size_depth = max(data_tokens_mapped.shape[0], self.fixed_size[0] + 2 * max_shift_depth)
            padded_size_height = max(data_tokens_mapped.shape[1], self.fixed_size[1] + 2 * max_shift_height)
            padded_size_width = max(data_tokens_mapped.shape[2], self.fixed_size[2] + 2 * max_shift_width)

            # Initialize padded data with air token
            padded_tokens_shape = (padded_size_depth, padded_size_height, padded_size_width)
            padded_tokens = np.full(padded_tokens_shape, self.air_token, dtype=np.int32)

            # Calculate offsets to place data_tokens_mapped into padded_tokens
            offset_depth = (padded_size_depth - data_tokens_mapped.shape[0]) // 2
            offset_height = (padded_size_height - data_tokens_mapped.shape[1]) // 2
            offset_width = (padded_size_width - data_tokens_mapped.shape[2]) // 2

            # Place data_tokens_mapped into padded_tokens
            padded_tokens[
                offset_depth:offset_depth + data_tokens_mapped.shape[0],
                offset_height:offset_height + data_tokens_mapped.shape[1],
                offset_width:offset_width + data_tokens_mapped.shape[2]
            ] = data_tokens_mapped

            # Now, select a random crop of size fixed_size from the padded data
            max_start_depth = padded_size_depth - self.fixed_size[0]
            max_start_height = padded_size_height - self.fixed_size[1]
            max_start_width = padded_size_width - self.fixed_size[2]

            # Random starting indices within valid range
            start_depth = np.random.randint(0, max_start_depth + 1)
            start_height = np.random.randint(0, max_start_height + 1)
            start_width = np.random.randint(0, max_start_width + 1)

            # Extract the crop
            data_tokens_mapped = padded_tokens[
                start_depth:start_depth + self.fixed_size[0],
                start_height:start_height + self.fixed_size[1],
                start_width:start_width + self.fixed_size[2]
            ]

        else:
            # === No Data Augmentation ===

            # Crop or pad data to fixed size
            crop_sizes = [min(data_tokens_mapped.shape[dim], self.fixed_size[dim]) for dim in range(3)]
            data_tokens_mapped = data_tokens_mapped[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2]]

            # Initialize padded data with the air token
            padded_tokens = np.full(self.fixed_size, self.air_token, dtype=np.int32)

            # Calculate offsets for centering the data
            offsets = [(self.fixed_size[dim] - crop_sizes[dim]) // 2 for dim in range(3)]
            slices_data = tuple(slice(offsets[dim], offsets[dim] + crop_sizes[dim]) for dim in range(3))

            # Place cropped data into the padded array
            padded_tokens[slices_data] = data_tokens_mapped

            data_tokens_mapped = padded_tokens

        # Convert tokens to torch tensor with dimensions (Depth, Height, Width)
        data_tokens_mapped = torch.from_numpy(data_tokens_mapped).long()

        return data_tokens_mapped
