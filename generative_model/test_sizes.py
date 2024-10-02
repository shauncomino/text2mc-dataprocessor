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
import torch.nn.functional as F

def log_cosh_loss(x, y, a=3.0):
    diff = a * (x - y)
    return torch.mean((1.0 / a) * torch.log(torch.cosh(diff)))

def loss_function(recon_x, x, mu, logvar, a=5.0):
    # If recon_x has more channels, slice to match x's channels
    recon_x = recon_x[:, :x.size(1), :, :, :]

    # Use log-cosh error instead of MSE
    log_cosh = log_cosh_loss(recon_x, x, a)

    # Calculate KL Divergence
    batch_size = x.size(0)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    return log_cosh + KLD

def test_batch_sizes(batch_sizes, fixed_size=(64, 64, 64)):
    # Paths and configurations
    tok2block_file_path = '/home/shaun/projects/text2mc-dataprocessor/world2vec/tok2block.json'
    block2embedding_file_path = '/home/shaun/projects/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'
    builds_folder_path = '/home/shaun/projects/text2mc-dataprocessor/test_builds'

    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device configuration
    device_type = 'cpu'
    device = torch.device(device_type)
    print("Using device:", device)

    # Load tok2block mapping
    with open(tok2block_file_path, 'r') as f:
        tok2block = json.load(f)
        tok2block = {int(k): v for k, v in tok2block.items()}  # Ensure keys are integers

    block2tok = {v: k for k, v in tok2block.items()}

    # Load block2embedding mapping
    with open(block2embedding_file_path, 'r') as f:
        block2embedding = json.load(f)
        # Convert embeddings to numpy arrays
        block2embedding = {k: np.array(v, dtype=np.float32) for k, v in block2embedding.items()}

    # Prepare the dataset
    hdf5_filepaths = glob.glob(os.path.join(builds_folder_path, '*.h5'))
    dataset = text2mcVAEDataset(
        file_paths=hdf5_filepaths,
        block2embedding=block2embedding,
        block2tok=block2tok,
        fixed_size=fixed_size
    )

    # Get embedding dimension from dataset
    embedding_dim = dataset.embedding_dim

    # Initialize the model components
    encoder = text2mcVAEEncoder().to(device)
    decoder = text2mcVAEDecoder().to(device)

    # Loop over different batch sizes
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        try:
            # Create DataLoader with the current batch size
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Get a single batch of data
            data_iter = iter(data_loader)
            data_batch = next(data_iter)
            data_batch = data_batch.to(device)

            print(f"Input data shape: {data_batch.shape}")

            # Pass data through the encoder
            with torch.no_grad():
                z, mu, logvar = encoder(data_batch)
                print(f"Encoder output (z) shape: {z.shape}")
                print(f"Encoder mu shape: {mu.shape}")
                print(f"Encoder logvar shape: {logvar.shape}")

                # Pass z through the decoder
                recon_batch = decoder(z)
                print(f"Decoder output shape: {recon_batch.shape}")

                # Optionally, compute the reconstruction loss
                recon_loss = loss_function(recon_batch, data_batch, mu, logvar)
                print(f"Reconstruction loss: {recon_loss.item()}")

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
