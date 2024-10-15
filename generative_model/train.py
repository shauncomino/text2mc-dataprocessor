# train.py

import torch
import torch.optim as optim
import glob
import json
import h5py
import os
from text2mcVAEDataset import text2mcVAEDataset
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from torch.utils.data import DataLoader
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

batch_size = 2
num_epochs = 64
fixed_size = (64, 64, 64)
embedding_dim = 32
on_arcc = True

if on_arcc:
    # Paths and configurations
    checkpoint_path = r'/lustre/fs1/home/scomino/training_lattice/checkpoint.pth'
    tok2block_file_path = r'/lustre/fs1/home/scomino/text2mc_lattice/text2mc-dataprocessor/world2vec/tok2block.json'
    builds_folder_path = r'/lustre/fs1/groups/jaedo/processed_builds'
    build1_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_319_8281.h5'
    build2_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_225_5840.h5'
    save_dir = r'/lustre/fs1/home/scomino/training_lattice/interpolations'
    best_model_path = r'/lustre/fs1/home/scomino/training_lattice/best_model.pth'
    block2embedding_file_path = r'/lustre/fs1/home/scomino/text2mc_lattice/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'
    
    # Device type for arcc
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    print("Using device:", device)
else:
    # Paths and configurations for local machine
    # Update these paths according to your local setup
    pass  # Replace with your local configurations

# Load mappings
with open(tok2block_file_path, 'r') as f:
    tok2block = json.load(f)
    tok2block = {int(k): v for k, v in tok2block.items()}

block2tok = {v: k for k, v in tok2block.items()}

with open(block2embedding_file_path, 'r') as f:
    block2embedding = json.load(f)
    block2embedding = {
        k: np.array(v, dtype=np.float32) for k, v in block2embedding.items()
    }

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Prepare the file paths
hdf5_filepaths = glob.glob(os.path.join(builds_folder_path, '*.h5'))
print(f"Discovered {len(hdf5_filepaths)} builds, beginning training")

# Split the file paths into training, validation, and test sets
dataset_size = len(hdf5_filepaths)
validation_split = 0.1
test_split = 0.1
train_size = int((1 - validation_split - test_split) * dataset_size)
val_size = int(validation_split * dataset_size)
test_size = dataset_size - train_size - val_size

# Shuffle file paths
random.shuffle(hdf5_filepaths)

# Split the file paths
train_file_paths = hdf5_filepaths[:train_size]
val_file_paths = hdf5_filepaths[train_size:train_size + val_size]
test_file_paths = hdf5_filepaths[train_size + val_size:]

# Create datasets
train_dataset = text2mcVAEDataset(
    file_paths=train_file_paths,
    block2tok=block2tok,
    block2embedding=block2embedding,
    fixed_size=fixed_size,
    augment=True
)

val_dataset = text2mcVAEDataset(
    file_paths=val_file_paths,
    block2tok=block2tok,
    block2embedding=block2embedding,
    fixed_size=fixed_size,
    augment=False
)

test_dataset = text2mcVAEDataset(
    file_paths=test_file_paths,
    block2tok=block2tok,
    block2embedding=block2embedding,
    fixed_size=fixed_size,
    augment=False
)

# Retrieve the air token ID
air_token_id = train_dataset.air_token
print(f"Air token ID: {air_token_id}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model components
encoder = text2mcVAEEncoder(embedding_dim=embedding_dim).to(device)
decoder = text2mcVAEDecoder(embedding_dim=embedding_dim).to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-5)

start_epoch = 1
best_val_loss = float('inf')

# Load checkpoint if it exists
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found, starting from scratch")

def loss_function(embeddings_pred, block_air_pred, x, mu, logvar, data_tokens, air_token_id, epsilon=1e-8):
    # embeddings_pred and x: (Batch_Size, Embedding_Dim, D, H, W)
    # block_air_pred: (Batch_Size, 1, D, H, W)
    # data_tokens: (Batch_Size, D, H, W)

    # Move Embedding_Dim to the last dimension
    embeddings_pred = embeddings_pred.permute(0, 2, 3, 4, 1)
    x = x.permute(0, 2, 3, 4, 1)

    # Flatten spatial dimensions
    batch_size, D, H, W, embedding_dim = embeddings_pred.shape
    N = batch_size * D * H * W
    embeddings_pred_flat = embeddings_pred.reshape(N, embedding_dim)
    x_flat = x.reshape(N, embedding_dim)

    # Flatten data_tokens
    data_tokens_flat = data_tokens.reshape(-1)

    # Prepare labels for Cosine Embedding Loss
    y = torch.ones(N, device=x_flat.device)

    # Compute Cosine Embedding Loss per voxel without reduction
    cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.0, reduction='none')
    loss_per_voxel = cosine_loss_fn(embeddings_pred_flat, x_flat, y)

    # Mask out the error tensor to include only the errors of the non-air blocks
    mask = (data_tokens_flat != air_token_id)
    loss_per_voxel = loss_per_voxel[mask]

    # Compute mean over non-air blocks
    num_non_air_voxels = loss_per_voxel.numel() + epsilon
    recon_loss = loss_per_voxel.sum() / num_non_air_voxels

    # Prepare ground truth labels for block vs. air
    block_air_labels = (data_tokens != air_token_id).float()

    # Compute binary cross-entropy loss
    bce_loss_fn = nn.BCELoss()
    block_air_pred = block_air_pred.squeeze(1)
    bce_loss = bce_loss_fn(block_air_pred, block_air_labels)

    # Compute KL Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine losses
    total_loss = recon_loss + bce_loss + KLD

    return total_loss, recon_loss, bce_loss, KLD

# Training loop
os.makedirs(save_dir, exist_ok=True)

for epoch in range(start_epoch, num_epochs + 1):
    encoder.train()
    decoder.train()
    average_reconstruction_loss = 0
    average_bce_loss = 0
    average_KL_divergence = 0

    for batch_idx, (data, data_tokens) in enumerate(train_loader):
        data = data.to(device)
        data_tokens = data_tokens.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        z, mu, logvar = encoder(data)
        embeddings_pred, block_air_pred = decoder(z)
        
        # Compute losses
        total_loss, recon_loss, bce_loss, KLD = loss_function(
            embeddings_pred, block_air_pred, data, mu, logvar, data_tokens, air_token_id=air_token_id
        )

        average_reconstruction_loss += recon_loss.item()
        average_bce_loss += bce_loss.item()
        average_KL_divergence += KLD.item()

        # Backward pass and optimization
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Logging
        if batch_idx % 30 == 0:
            print(f'Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)}] '
                  f'Recon Loss: {recon_loss.item():.6f}, KL Divergence: {KLD.item():.6f}, '
                  f'BCE Loss: {bce_loss.item():.6f}')

    # Compute average losses over the epoch
    average_reconstruction_loss /= len(train_loader)
    average_bce_loss /= len(train_loader)
    average_KL_divergence /= len(train_loader)
    print(f'====> Epoch: {epoch} Average Reconstruction Loss: {average_reconstruction_loss:.8f}')
    print(f'====> Epoch: {epoch} Average KL-Divergence: {average_KL_divergence:.8f}')
    print(f'====> Epoch: {epoch} Average BCE Loss: {average_bce_loss:.8f}')

    # Validation
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for data, data_tokens in val_loader:
            data = data.to(device)
            data_tokens = data_tokens.to(device)
            z, mu, logvar = encoder(data)
            embeddings_pred, block_air_pred = decoder(z)
            total_loss, recon_loss, bce_loss, KLD = loss_function(
                embeddings_pred, block_air_pred, data, mu, logvar, data_tokens, air_token_id=air_token_id
            )
            val_loss += total_loss.item()

    val_loss /= len(val_loader)
    print(f'====> Epoch: {epoch} Validation Loss: {val_loss:.4f}')

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, best_model_path)
        print(f'Saved new best model at epoch {epoch} with validation loss {val_loss:.4f}')

    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'seed': seed,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint at epoch {epoch}")
