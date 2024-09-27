import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class text2mcVAEDataset(Dataset):
    def __init__(self, file_paths=[], block2tok={}, block_ignore_list=[], fixed_size=(64, 64, 64)):
        self.file_paths = file_paths
        self.block2tok = block2tok
        self.block_ignore_set = set(block_ignore_list)
        self.fixed_size = fixed_size

        # Prepare token mappings
        self.prepare_token_mappings()

    def prepare_token_mappings(self):
        # Convert block names to tokens (integers)
        self.block2token_int = {block_name: int(token_str) for block_name, token_str in self.block2tok.items()}

        # Get the air token
        air_block_name = 'minecraft:air'
        air_token_str = self.block2tok.get(air_block_name)
        if air_token_str is None:
            raise ValueError('minecraft:air block not found in block2tok mapping')
        self.air_token = int(air_token_str)

        # Collect all tokens
        tokens = list(self.block2token_int.values())

        # Ensure tokens are non-negative integers
        if min(tokens) < 0:
            raise ValueError("All block tokens must be non-negative integers.")

        self.token_set = set(tokens)

        # Find the maximum token value to determine the size of the lookup arrays
        self.max_token = max(tokens + [3714])  # Include unknown block token

        # Build the lookup array mapping tokens to indices
        self.lookup_array = np.full((self.max_token + 1,), self.air_token, dtype=np.int32)
        for block_name, token in self.block2token_int.items():
            self.lookup_array[token] = token  # Map token to itself

        # Map unknown block token to 3714
        self.unknown_token = 3714
        self.lookup_array[self.unknown_token] = self.unknown_token

        # For tokens in block_ignore_set, map them to air_token
        for block_name in self.block_ignore_set:
            token_str = self.block2tok.get(block_name)
            if token_str is not None:
                token = int(token_str)
                if token <= self.max_token:
                    self.lookup_array[token] = self.air_token

        self.num_tokens = self.max_token + 1  # Total number of tokens including unknown and air

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
            self.unknown_token  # Assign unknown block token
        )

        # Map tokens using the lookup array
        indices_array = self.lookup_array[data_tokens_mapped]

        # Create mask: 1 where token is not air_token, else 0
        mask = np.where(indices_array != self.air_token, 1.0, 0.0)

        # Crop or pad data and mask to fixed size
        crop_sizes = [min(indices_array.shape[dim], self.fixed_size[dim]) for dim in range(3)]
        indices_array = indices_array[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2]]
        mask = mask[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2]]

        # Initialize padded data and mask
        padded_data = np.full(self.fixed_size, self.air_token, dtype=np.int32)
        padded_mask = np.zeros(self.fixed_size, dtype=np.float32)

        # Calculate offsets for centering the data
        offsets = [(self.fixed_size[dim] - crop_sizes[dim]) // 2 for dim in range(3)]
        slices_data = tuple(slice(offsets[dim], offsets[dim] + crop_sizes[dim]) for dim in range(3))

        # Place cropped data into the padded arrays
        padded_data[slices_data] = indices_array
        padded_mask[slices_data] = mask

        # Convert to torch tensors
        padded_data = torch.from_numpy(padded_data).long()  # Use LongTensor for indices
        padded_mask = torch.from_numpy(padded_mask)

        return padded_data, padded_mask
