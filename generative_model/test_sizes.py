import torch
import glob
import json
import os
import numpy as np
import random

from torch.utils.data import DataLoader
from text2mcVAEDataset import text2mcVAEDataset
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder

def test_batch_sizes(batch_sizes, fixed_size=(64, 64, 64)):
    # Paths and configurations
    block2tok_file_path = '/home/shaun/projects/text2mc-dataprocessor/world2vec/tok2block.json'
    builds_folder_path = '/home/shaun/projects/text2mc-dataprocessor/test_builds'

    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device configuration
    # device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cpu'
    device = torch.device(device_type)
    print("Using device:", device)

    # Load block2tok
    with open(block2tok_file_path, 'r') as j:
        block2tok = json.load(j)
        block2tok = dict((v, k) for k, v in block2tok.items())

    # Prepare the dataset
    hdf5_filepaths = glob.glob(os.path.join(builds_folder_path, '*.h5'))
    dataset = text2mcVAEDataset(file_paths=hdf5_filepaths, block2tok=block2tok, fixed_size=fixed_size)

    # Get num_tokens from dataset
    num_tokens = dataset.num_tokens  # Total number of tokens
    embedding_dim = 32  # Choose an embedding dimension

    # Initialize the model components
    encoder = text2mcVAEEncoder(num_tokens=num_tokens, embedding_dim=embedding_dim).to(device)
    decoder = text2mcVAEDecoder(num_tokens=num_tokens, embedding_dim=embedding_dim).to(device)

    # Loop over different batch sizes
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        try:
            # Create DataLoader with the current batch size
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Get a single batch of data
            data_iter = iter(data_loader)
            data_batch, mask_batch = next(data_iter)
            data_batch, mask_batch = data_batch.to(device), mask_batch.to(device)

            print(f"Input data shape: {data_batch.shape}")
            print(f"Mask shape: {mask_batch.shape}")

            # Pass data through the encoder
            with torch.no_grad():
                z, mu, logvar = encoder(data_batch)
                print(f"Encoder output (z) shape: {z.shape}")
                print(f"Encoder mu shape: {mu.shape}")
                print(f"Encoder logvar shape: {logvar.shape}")

                # Pass z through the decoder
                recon_batch = decoder(z)
                print(f"Decoder output shape: {recon_batch.shape}")

            # Optionally, print more details about the model
            # print(encoder)
            # print(decoder)

        except RuntimeError as e:
            print(f"RuntimeError for batch size {batch_size}: {e}")
        except Exception as e:
            print(f"Exception for batch size {batch_size}: {e}")

if __name__ == '__main__':
    # Define the batch sizes you want to test
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # Optionally, define the fixed size for input data
    fixed_size = (64, 64, 64)  # Adjust as needed

    # Call the testing function
    test_batch_sizes(batch_sizes, fixed_size=fixed_size)
