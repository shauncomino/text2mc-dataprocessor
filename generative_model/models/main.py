# main.py

import torch
from config import Config
from train import train
from text2mcVAEDataset import text2mcVAEDataset
import glob
import os
import random
import numpy as np
import json


def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Paths and configurations
    builds_folder_path = '/path/to/builds/'
    tok2block_file_path = '/path/to/tok2block.json'
    block2embedding_file_path = '/path/to/embeddings.json'
    save_dir = '/path/to/save_dir'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load mappings
    with open(tok2block_file_path, 'r') as f:
        tok2block = json.load(f)
        tok2block = {int(k): v for k, v in tok2block.items()}  # Ensure keys are integers

    block2tok = {v: k for k, v in tok2block.items()}

    with open(block2embedding_file_path, 'r') as f:
        block2embedding = json.load(f)
        # Load embeddings
        block2embedding = {
            k: np.array(v, dtype=np.float32) for k, v in block2embedding.items()
        }

    # Prepare the file paths
    hdf5_filepaths = glob.glob(os.path.join(builds_folder_path, '*.h5'))
    print(f"Discovered {len(hdf5_filepaths)} builds, beginning training")

    # Shuffle file paths
    random.shuffle(hdf5_filepaths)

    # Create dataset
    dataset = text2mcVAEDataset(
        file_paths=hdf5_filepaths,
        block2embedding=block2embedding,
        block2tok=block2tok,
        fixed_size=(64, 64, 64),
        augment=False
    )

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Get one sample
    for data, _ in dataloader:
        real = data  # Embedded data
        break  # Use only one sample for now

    real = real.to(device)

    # Set up config
    opt = Config()
    opt.device = device
    opt.level_shape = real.shape  # Should be (Batch_Size, Channels, D, H, W)
    opt.nc_current = real.shape[1]  # Number of channels
    opt.out_ = save_dir

    # Start training
    train(real, opt)


if __name__ == "__main__":
    main()
