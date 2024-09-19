import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import time
import matplotlib.pyplot as plt
import json

# Original Data Loader with Nested Loops
class OriginalText2mcVAEDataset(Dataset):
    def __init__(self, file_paths=[], block2embedding={}, block2tok={}, block_ignore_list=[], fixed_size=(256, 256, 256, 32)):
        self.file_paths = file_paths
        self.block2embedding = block2embedding
        self.block2tok = block2tok
        self.block_ignore_list = set(block_ignore_list)
        self.fixed_size = fixed_size

        # Prepare the embeddings and mappings
        self.prepare_embeddings()

    def prepare_embeddings(self):
        # Build tok2embedding mapping
        self.tok2embedding = {}
        for block_name, embedding in self.block2embedding.items():
            token = self.block2tok.get(block_name)
            if token is not None:
                self.tok2embedding[int(token)] = np.array(embedding, dtype=np.float32)
        
        # Get the air token and embedding
        air_block_name = 'minecraft:air'
        air_token = self.block2tok.get(air_block_name)
        if air_token is None:
            raise ValueError('minecraft:air block not found in block2tok mapping')
        self.air_token = int(air_token)
        air_embedding = self.block2embedding.get(air_block_name)
        if air_embedding is None:
            raise ValueError('minecraft:air block not found in block2embedding mapping')
        self.air_embedding = np.array(air_embedding, dtype=np.float32)

        # Ensure that the air token and embedding are included
        self.tok2embedding[self.air_token] = self.air_embedding

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with h5py.File(file_path, 'r') as file:
            build_folder_in_hdf5 = list(file.keys())[0]
            data = file[build_folder_in_hdf5][()]
        
        # Embed and create mask
        embedded_data = np.zeros((*data.shape, len(self.air_embedding)), dtype=np.float32)
        mask = np.zeros(data.shape, dtype=np.float32)
        
        data_shape = data.shape
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                for k in range(data_shape[2]):
                    block_token = data[i, j, k]
                    # Handle unknown or ignored blocks
                    if block_token in self.tok2embedding and block_token not in self.block_ignore_list:
                        embedded_data[i, j, k] = self.tok2embedding[block_token]
                        mask[i, j, k] = 1
                    else:
                        # Replace with air embedding
                        embedded_data[i, j, k] = self.air_embedding
                        mask[i, j, k] = 0

        # Calculate the cropping or padding
        crop_sizes = [min(data_shape[dim], self.fixed_size[dim]) for dim in range(3)]
        embedded_data = embedded_data[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2], :]
        mask = mask[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2]]

        # Pad the data and the mask to fixed size
        padded_data = np.zeros(self.fixed_size, dtype=np.float32)
        padded_mask = np.zeros(self.fixed_size[:3], dtype=np.float32)

        offsets = [(self.fixed_size[dim] - crop_sizes[dim]) // 2 for dim in range(3)]
        slices_data = tuple(slice(offsets[dim], offsets[dim] + crop_sizes[dim]) for dim in range(3))

        padded_data[slices_data] = embedded_data
        padded_mask[slices_data] = mask

        padded_data = torch.from_numpy(padded_data).permute(3, 0, 1, 2)
        padded_mask = torch.from_numpy(padded_mask)

        return padded_data, padded_mask

# Optimized Data Loader with NumPy Indexing
class OptimizedText2mcVAEDataset(Dataset):
    def __init__(self, file_paths=[], block2embedding={}, block2tok={}, block_ignore_list=[], fixed_size=(256, 256, 256, 32)):
        self.file_paths = file_paths
        self.block2embedding = block2embedding
        self.block2tok = block2tok
        self.block_ignore_set = set(block_ignore_list)
        self.fixed_size = fixed_size

        # Prepare the embedding matrix and lookup arrays
        self.prepare_embedding_matrix()

    def prepare_embedding_matrix(self):
        # Build tok2embedding mapping
        self.tok2embedding_int = {}
        for block_name, embedding in self.block2embedding.items():
            token = self.block2tok.get(block_name)
            if token is not None:
                self.tok2embedding_int[int(token)] = np.array(embedding, dtype=np.float32)
        
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
        self.embedding_matrix = np.zeros((self.max_token + 1,), dtype=object)
        for token in range(self.max_token + 1):
            if token in self.tok2embedding_int:
                self.embedding_matrix[token] = self.tok2embedding_int[token]
            else:
                # Replace with air embedding
                self.embedding_matrix[token] = self.air_embedding

        # Build the valid tokens array for creating the mask
        self.valid_tokens_array = np.ones((self.max_token + 1,), dtype=np.bool_)
        self.valid_tokens_array[self.air_token] = False  # Air block is not valid
        # Set ignored blocks to False
        for block_name in self.block_ignore_set:
            token_str = self.block2tok.get(block_name)
            if token_str is not None:
                token = int(token_str)
                if token <= self.max_token:
                    self.valid_tokens_array[token] = False

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

        # Retrieve embeddings using indices
        # Shape: (Depth, Height, Width, Embedding_Dim)
        embedded_data = np.empty(data_tokens_mapped.shape + (self.embedding_dim,), dtype=np.float32)
        for idx_block, token in enumerate(np.unique(data_tokens_mapped)):
            mask_token = data_tokens_mapped == token
            embedded_data[mask_token] = self.embedding_matrix[token]

        # Create mask: 1 where token is valid and not in block_ignore_set, else 0
        mask = self.valid_tokens_array[data_tokens_mapped].astype(np.float32)

        # Crop or pad data and mask to fixed size
        crop_sizes = [min(embedded_data.shape[dim], self.fixed_size[dim]) for dim in range(3)]
        embedded_data = embedded_data[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2], :]
        mask = mask[:crop_sizes[0], :crop_sizes[1], :crop_sizes[2]]

        # Initialize padded data and mask
        padded_data = np.zeros(self.fixed_size, dtype=np.float32)
        padded_mask = np.zeros(self.fixed_size[:3], dtype=np.float32)

        # Calculate offsets for centering the data
        offsets = [(self.fixed_size[dim] - crop_sizes[dim]) // 2 for dim in range(3)]
        slices_data = tuple(slice(offsets[dim], offsets[dim] + crop_sizes[dim]) for dim in range(3))

        # Place cropped data into the padded arrays
        padded_data[slices_data] = embedded_data
        padded_mask[slices_data] = mask

        # Convert to torch tensors and permute dimensions to (Embedding_Dim, Depth, Height, Width)
        padded_data = torch.from_numpy(padded_data).permute(3, 0, 1, 2)
        padded_mask = torch.from_numpy(padded_mask)

        return padded_data, padded_mask

# Function to measure execution time
def measure_execution_time(dataset_class, file_paths, block2embedding, block2tok, block_ignore_list, fixed_size, num_trials=5):
    times = []
    for trial in range(num_trials):
        start_time = time.perf_counter()
        dataset = dataset_class(file_paths, block2embedding, block2tok, block_ignore_list, fixed_size)
        # Simulate loading all data
        for idx in range(len(dataset)):
            data, mask = dataset[idx]
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        print(f"Trial {trial + 1}/{num_trials}: {elapsed_time:.4f} seconds")
    average_time = np.mean(times)
    std_time = np.std(times)
    return average_time, std_time

# Main testing code
if __name__ == "__main__":
    # File paths to HDF5 files
    hdf5_filepaths = [
        r'generative_model/rar_test6_Desert_Tavern.h5',
        r'generative_model/rar_test5_Desert+Tavern+2.h5',
        r'generative_model/zip_test_0_LargeSandDunes.h5'
    ]

    # Load the real JSON dictionaries
    with open('block2vec/output/block2vec/embeddings.json', 'r') as f:
        block2embedding = json.load(f)

    with open('world2vec/block2tok.json', 'r') as f:
        block2tok = json.load(f)

    # Block ignore list
    block_ignore_list = ['minecraft:grass_block']

    # Fixed size
    embedding_dim = len(next(iter(block2embedding.values())))
    fixed_size = (256, 256, 256, embedding_dim)

    # Number of trials
    num_trials = 50

    # Measure execution time for the original data loader
    print("Testing Original Data Loader with Nested Loops...")
    avg_time_original, std_time_original = measure_execution_time(
        OriginalText2mcVAEDataset,
        hdf5_filepaths,
        block2embedding,
        block2tok,
        block_ignore_list,
        fixed_size,
        num_trials
    )

    # Measure execution time for the optimized data loader
    print("\nTesting Optimized Data Loader with NumPy Indexing...")
    avg_time_optimized, std_time_optimized = measure_execution_time(
        OptimizedText2mcVAEDataset,
        hdf5_filepaths,
        block2embedding,
        block2tok,
        block_ignore_list,
        fixed_size,
        num_trials
    )

    # Plotting the results
    methods = ['Original', 'Optimized']
    avg_times = [avg_time_original, avg_time_optimized]
    std_times = [std_time_original, std_time_optimized]

    plt.figure(figsize=(8, 6))
    plt.bar(methods, avg_times, yerr=std_times, capsize=10, color=['red', 'green'])
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Triple Loop vs Vectorized Method')
    plt.grid(axis='y')
    plt.show()
