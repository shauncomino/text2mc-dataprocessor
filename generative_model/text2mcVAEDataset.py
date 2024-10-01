import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class text2mcVAEDataset(Dataset):
    def __init__(self, file_paths=[], block2embedding={}, block2tok={}, block_ignore_list=[], fixed_size=(64, 64, 64)):
        self.file_paths = file_paths
        self.block2embedding = block2embedding
        self.block2tok = block2tok
        self.block_ignore_set = set(block_ignore_list)
        self.fixed_size = fixed_size  # (Depth, Height, Width)

        # Prepare the embedding matrix and lookup arrays
        self.prepare_embedding_matrix()

    def prepare_embedding_matrix(self):
        # Build token to embedding mapping
        self.tok2embedding_int = {}
        for block_name, embedding in self.block2embedding.items():
            token_str = self.block2tok.get(block_name)
            if token_str is not None:
                token = int(token_str)
                self.tok2embedding_int[token] = np.array(embedding, dtype=np.float32)

        # Get the air token and embedding
        air_block_name = 'minecraft:air'
        air_token_str = self.block2tok.get(air_block_name)
        if air_token_str is None:
            raise ValueError('minecraft:air block not found in block2tok mapping')
        self.air_token = int(air_token_str)
        air_embedding = self.block2embedding.get(air_block_name)
        if air_embedding is None:
            raise ValueError('minecraft:air block not found in block2embedding mapping')
        self.air_embedding = np.array(air_embedding, dtype=np.float32)

        # Ensure that the air token and embedding are included
        self.tok2embedding_int[self.air_token] = self.air_embedding

        # Collect all tokens
        tokens = list(self.tok2embedding_int.keys())

        # Ensure tokens are non-negative integers
        if min(tokens) < 0:
            raise ValueError("All block tokens must be non-negative integers.")

        self.token_set = set(tokens)
        self.embedding_dim = len(self.air_embedding)

        # Find the maximum token value to determine the size of the lookup arrays
        self.max_token = max(tokens + [3714])  # Include unknown block token

        # Build the embedding matrix
        self.embedding_matrix = np.zeros((self.max_token + 1, self.embedding_dim), dtype=np.float32)
        for token in tokens:
            self.embedding_matrix[token] = self.tok2embedding_int[token]
        # Set the embedding for unknown blocks to the air embedding
        self.embedding_matrix[3714] = self.air_embedding  # Token 3714 is the unknown block

        # Build the lookup array mapping tokens to indices in the embedding matrix
        self.lookup_array = np.full((self.max_token + 1,), self.air_token, dtype=np.int32)
        for token in tokens:
            self.lookup_array[token] = token  # Map token to its own index

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

        # Map tokens to indices in the embedding matrix using the lookup array
        indices_array = self.lookup_array[data_tokens_mapped]

        # Retrieve embeddings using indices
        # Shape: (Depth, Height, Width, Embedding_Dim)
        embedded_data = self.embedding_matrix[indices_array]

        # Crop or pad data to fixed size
        crop_sizes = [min(embedded_data.shape[dim], self.fixed_size[dim]) for dim in range(3)]
        embedded_data = embedded_data[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2], :]

        # Initialize padded data with the air embedding
        padded_data = np.empty((*self.fixed_size, self.embedding_dim), dtype=np.float32)
        padded_data[...] = self.air_embedding

        # Calculate offsets for centering the data
        offsets = [(self.fixed_size[dim] - crop_sizes[dim]) // 2 for dim in range(3)]
        slices_data = tuple(slice(offsets[dim], offsets[dim] + crop_sizes[dim]) for dim in range(3))

        # Place cropped data into the padded array
        padded_data[slices_data] = embedded_data

        # Convert to torch tensor and permute dimensions to (Embedding_Dim, Depth, Height, Width)
        padded_data = torch.from_numpy(padded_data).permute(3, 0, 1, 2)

        return padded_data
