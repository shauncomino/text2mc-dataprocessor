# train.py

import torch
import torch.optim as optim
import glob
import json
import h5py
import os
from dataloader import text2mcVAEDataset
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import math

batch_size = 2
num_epochs = 64
fixed_size = (64, 64, 64)
embedding_dim = 32

# Paths and configurations
checkpoint_path = r'/lustre/fs1/home/scomino/training/checkpoint.pth'
tok2block_file_path = r'/lustre/fs1/home/scomino/text2mc/text2mc-dataprocessor/world2vec/tok2block.json'
builds_folder_path = r'/lustre/fs1/groups/jaedo/processed_builds'
build1_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_101_2606.h5'
build2_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_118_3048.h5'
save_dir = r'/lustre/fs1/home/scomino/training/interpolations'
best_model_path = r'/lustre/fs1/home/scomino/training/best_model.pth'
block2embedding_file_path = r'/lustre/fs1/home/scomino/text2mc/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'

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

# Device type
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
print("Using device:", device)

# Prepare the dataset
hdf5_filepaths = glob.glob(os.path.join(builds_folder_path, '*.h5'))
dataset = text2mcVAEDataset(
    file_paths=hdf5_filepaths,
    block2tok=block2tok,
    block2embedding=block2embedding,
    fixed_size=fixed_size
)

print(f"Discovered {len(dataset)} builds, beginning training")

# Retrieve the air token ID
air_token_id = dataset.air_token
print(f"Air token ID: {air_token_id}")

# Split the dataset into training, validation, and test sets
dataset_size = len(dataset)
validation_split = 0.2
test_split = 0.1
train_size = int((1 - validation_split - test_split) * dataset_size)
val_size = int(validation_split * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(seed)
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

# Define the loss function with masking for air blocks
def loss_function(recon_x, x, mu, logvar, data_tokens, air_token_id, a=5.0):
    # If recon_x has more channels, slice to match x's channels
    recon_x = recon_x[:, :x.size(1), :, :, :]

    # Compute the per-voxel difference using log-cosh loss
    diff = a * (recon_x - x)
    loss = (1.0 / a) * (diff + F.softplus(-2.0 * diff) - math.log(2.0))

    # Create a mask where the token is not air
    mask = (data_tokens != air_token_id).unsqueeze(1).float()  # Shape: (Batch_Size, 1, D, H, W)

    # Apply the mask to the loss
    masked_loss = loss * mask

    # Compute the mean of the masked loss
    recon_loss = torch.mean(masked_loss)

    # Calculate KL Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + KLD

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
            augment=True
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

            # Save the build as an HDF5 file
            save_path = os.path.join(save_dir, f'epoch_{epoch}_interp_{idx}.h5')
            with h5py.File(save_path, 'w') as h5f:
                h5f.create_dataset('build', data=recon_tokens_np, compression='gzip')

            print(f'Saved interpolated build at {save_path}')

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
