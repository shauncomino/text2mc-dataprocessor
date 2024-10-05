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
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import torch.nn as nn
import math

batch_size = 2
num_epochs = 64
fixed_size = (64, 64, 64)
embedding_dim = 32
on_arcc = True

if on_arcc:
    # Paths and configurations
    checkpoint_path = r'/lustre/fs1/home/scomino/training/checkpoint.pth'
    tok2block_file_path = r'/lustre/fs1/home/scomino/text2mc/text2mc-dataprocessor/world2vec/tok2block.json'
    builds_folder_path = r'/lustre/fs1/groups/jaedo/processed_builds'
    build1_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_101_2606.h5'
    build2_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_118_3048.h5'
    save_dir = r'/lustre/fs1/home/scomino/training/interpolations'
    best_model_path = r'/lustre/fs1/home/scomino/training/best_model.pth'
    block2embedding_file_path = r'/lustre/fs1/home/scomino/text2mc/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'
    
    # Device type for arcc
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    print("Using device:", device)
else:
    builds_folder_path = r'/home/shaun/projects/text2mc-dataprocessor/test_builds/'
    tok2block_file_path = r'/home/shaun/projects/text2mc-dataprocessor/world2vec/tok2block.json'
    block2embedding_file_path = r'/home/shaun/projects/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'
    checkpoint_path = r'/home/shaun/projects/text2mc-dataprocessor/checkpoint.pth'
    best_model_path = r'/home/shaun/projects/text2mc-dataprocessor/best_model.pth'
    build1_path = r'/home/shaun/projects/text2mc-dataprocessor/test_builds/batch_157_4066.h5'
    build2_path = r'/home/shaun/projects/text2mc-dataprocessor/test_builds/batch_157_4077.h5'
    save_dir = r'/home/shaun/projects/text2mc-dataprocessor/test_builds/'

    # Device type for local machine
    device_type = 'cpu'
    device = torch.device(device_type)
    print("Using device:", device)


# Load mappings
with open(tok2block_file_path, 'r') as f:
    tok2block = json.load(f)
    tok2block = {int(k): v for k, v in tok2block.items()}  # Ensure keys are integers

block2tok = {v: k for k, v in tok2block.items()}

with open(block2embedding_file_path, 'r') as f:
    block2embedding = json.load(f)
    block2embedding = {k: np.array(v, dtype=np.float32) for k, v in block2embedding.items()}

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
validation_split = 0.2
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

# Create separate datasets with appropriate augmentation settings
train_dataset = text2mcVAEDataset(
    file_paths=train_file_paths,
    block2tok=block2tok,
    block2embedding=block2embedding,
    fixed_size=fixed_size,
    augment=True  # Enable augmentations for training
)

val_dataset = text2mcVAEDataset(
    file_paths=val_file_paths,
    block2tok=block2tok,
    block2embedding=block2embedding,
    fixed_size=fixed_size,
    augment=False  # Disable augmentations for validation
)

test_dataset = text2mcVAEDataset(
    file_paths=test_file_paths,
    block2tok=block2tok,
    block2embedding=block2embedding,
    fixed_size=fixed_size,
    augment=False  # Disable augmentations for testing
)

# Retrieve the air token ID from the training dataset
air_token_id = train_dataset.air_token
print(f"Air token ID: {air_token_id}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model components
encoder = text2mcVAEEncoder().to(device)
decoder = text2mcVAEDecoder().to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-5, weight_decay=1e-3)

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

# Initialize CosineEmbeddingLoss
cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.0, reduction='sum')

def loss_function(recon_x, x, mu, logvar, data_tokens, air_token_id, epsilon=1e-6):
    # recon_x and x are both of shape (Batch_Size, Embedding_Dim, D, H, W)
    # We need to reshape them to compute the loss per voxel
    
    # Move Embedding_Dim to the last dimension
    recon_x = recon_x.permute(0, 2, 3, 4, 1)  # Shape: (Batch_Size, D, H, W, Embedding_Dim)
    x = x.permute(0, 2, 3, 4, 1)              # Shape: (Batch_Size, D, H, W, Embedding_Dim)
    
    # Flatten the spatial dimensions
    recon_x_flat = recon_x.reshape(-1, recon_x.shape[-1])  # Shape: (N, Embedding_Dim)
    x_flat = x.reshape(-1, x.shape[-1])                    # Shape: (N, Embedding_Dim)
    
    # Create a mask where the token is not air
    mask = (data_tokens != air_token_id)  # Shape: (Batch_Size, D, H, W)
    mask_flat = mask.reshape(-1)          # Shape: (N,)
    
    # Filter out air voxels
    recon_x_flat = recon_x_flat[mask_flat]
    x_flat = x_flat[mask_flat]
    
    # Prepare the labels (y = 1 for similar embeddings)
    y = torch.ones(x_flat.size(0)).to(x_flat.device)
    
    # Compute the Cosine Embedding Loss
    recon_loss = cosine_loss_fn(recon_x_flat, x_flat, y)
    
    # Normalize the loss by the number of non-air voxels
    num_non_air_voxels = x_flat.size(0) + epsilon
    recon_loss = recon_loss / num_non_air_voxels
    
    # KL Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + KLD



# Function to convert embeddings back to tokens
def embedding_to_tokens(embedded_data, embeddings_matrix):
    # embedded_data: PyTorch tensor of shape (Batch_Size, Embedding_Dim, Depth, Height, Width)
    # embeddings_matrix: NumPy array or PyTorch tensor of shape (Num_Tokens, Embedding_Dim)

    batch_size, embedding_dim, D, H, W = embedded_data.shape

    # Convert embedded_data to NumPy array
    embedded_data_np = embedded_data.detach().cpu().numpy()

    # Ensure embeddings_matrix is a NumPy array
    if isinstance(embeddings_matrix, torch.Tensor):
        embeddings_matrix_np = embeddings_matrix.detach().cpu().numpy()
    else:
        embeddings_matrix_np = embeddings_matrix

    # Flatten the embedded data
    N = D * H * W
    embedded_data_flat = embedded_data_np.reshape(batch_size, embedding_dim, N)
    embedded_data_flat = embedded_data_flat.transpose(0, 2, 1)  # Shape: (Batch_Size, N, Embedding_Dim)
    embedded_data_flat = embedded_data_flat.reshape(-1, embedding_dim)  # Shape: (Batch_Size * N, Embedding_Dim)

    # Initialize NearestNeighbors with Euclidean distance
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
    nbrs.fit(embeddings_matrix_np)

    # Find nearest neighbors
    distances, indices = nbrs.kneighbors(embedded_data_flat)
    tokens_flat = indices.flatten()  # Shape: (Batch_Size * N,)

    # Reshape tokens back to (Batch_Size, Depth, Height, Width)
    tokens = tokens_flat.reshape(batch_size, D, H, W)
    tokens = torch.from_numpy(tokens).long()  # Convert to torch tensor

    return tokens

# Function to interpolate and generate builds
def interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, epoch, num_interpolations=5):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Load the two builds
        dataset = text2mcVAEDataset(
            file_paths=[build1_path, build2_path],
            block2tok=block2tok,
            block2embedding=block2embedding,
            fixed_size=fixed_size,
            augment=False  # Disable augmentation for consistent reconstructions
        )
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        data_list = []
        # We aren't using the tokens directly in this case
        for data, _ in data_loader:
            data = data.to(device)
            data_list.append(data)

        z_list = []
        for data in data_list:
            z, mu, logvar = encoder(data)
            z_list.append(z)

        # Generate reconstructions of the original builds
        for idx, (data, z) in enumerate(zip(data_list, z_list)):
            recon_embedded = decoder(z)
            # Convert embeddings back to tokens
            recon_tokens = embedding_to_tokens(recon_embedded, dataset.embedding_matrix)
            # Convert to numpy array
            recon_tokens_np = recon_tokens.cpu().numpy().squeeze(0)  # Shape: (Depth, Height, Width)

            # Save the reconstructed build as an HDF5 file
            save_path = os.path.join(save_dir, f'epoch_{epoch}_recon_{idx}.h5')
            with h5py.File(save_path, 'w') as h5f:
                h5f.create_dataset('build', data=recon_tokens_np, compression='gzip')

            print(f'Saved reconstruction of build {idx + 1} at {save_path}')

        # Interpolate between z1 and z2
        z1 = z_list[0]
        z2 = z_list[1]

        interpolations = []
        for alpha in np.linspace(0, 1, num_interpolations):
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolations.append(z_interp)

        # Generate builds from interpolated latent vectors
        for idx, z in enumerate(interpolations):
            recon_embedded = decoder(z)
            # recon_embedded: (1, Embedding_Dim, Depth, Height, Width)

            # Convert embeddings back to tokens
            recon_tokens = embedding_to_tokens(recon_embedded, dataset.embedding_matrix)
            # recon_tokens: (1, Depth, Height, Width)

            # Convert to numpy array
            recon_tokens_np = recon_tokens.cpu().numpy().squeeze(0)  # Shape: (Depth, Height, Width)

            # Save the interpolated build as an HDF5 file
            save_path = os.path.join(save_dir, f'epoch_{epoch}_interp_{idx}.h5')
            with h5py.File(save_path, 'w') as h5f:
                h5f.create_dataset('build', data=recon_tokens_np, compression='gzip')

            print(f'Saved interpolated build at {save_path}')

# Training loop
os.makedirs(save_dir, exist_ok=True)

for epoch in range(start_epoch, num_epochs + 1):
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch_idx, (data, data_tokens) in enumerate(train_loader):
        data = data.to(device)          # Embedded data
        data_tokens = data_tokens.to(device)  # Tokens
        optimizer.zero_grad()
        
        z, mu, logvar = encoder(data)
        recon_batch = decoder(z)
        loss = loss_function(recon_batch, data, mu, logvar, data_tokens, air_token_id)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}')

    avg_train_loss = total_loss / len(train_loader)
    print(f'====> Epoch: {epoch} Average training loss: {avg_train_loss:.4f}')

    # Validation
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for data, data_tokens in val_loader:
            data = data.to(device)
            data_tokens = data_tokens.to(device)
            
            z, mu, logvar = encoder(data)
            recon_batch = decoder(z)
            loss = loss_function(recon_batch, data, mu, logvar, data_tokens, air_token_id)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'====> Epoch: {epoch} Validation loss: {avg_val_loss:.4f}')

    # Check if validation loss improved, and save model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, best_model_path)
        print(f'Saved new best model at epoch {epoch} with validation loss {avg_val_loss:.4f}')

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

    # Interpolate and generate builds
    print(f'Interpolating between builds at the end of epoch {epoch}')
    try:
        interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, epoch, num_interpolations=5)
    except Exception as e:
        print(f"Unable to generate interpolations for this epoch due to error: {e}")

