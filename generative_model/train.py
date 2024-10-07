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
embedding_dim = 32  # Adjusted embedding dimension
on_arcc = True

if on_arcc:
    # Paths and configurations
    checkpoint_path = r'/lustre/fs1/home/scomino/training_genembed/checkpoint.pth'
    best_model_path = r'/lustre/fs1/home/scomino/training_genembed/best_model.pth'
    save_dir = r'/lustre/fs1/home/scomino/training_genembed/interpolations'
    tok2block_file_path = r'/lustre/fs1/home/scomino/text2mc_genembed/text2mc-dataprocessor/world2vec/tok2block.json'

    builds_folder_path = r'/lustre/fs1/groups/jaedo/processed_builds'
    build1_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_319_8281.h5'
    build2_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_225_5840.h5'
    
    
    
    # Device type for arcc
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    print("Using device:", device)
else:
    builds_folder_path = r'/home/shaun/projects/text2mc-dataprocessor/test_builds/'
    tok2block_file_path = r'/home/shaun/projects/text2mc-dataprocessor/world2vec/tok2block.json'
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

def loss_function(recon_logits, data_tokens, mu, logvar, air_token_id):
    # recon_logits: (Batch_Size, num_tokens, D, H, W)
    # data_tokens: (Batch_Size, D, H, W)

    # Flatten spatial dimensions
    batch_size, num_tokens, D, H, W = recon_logits.shape
    N = batch_size * D * H * W

    recon_logits_flat = recon_logits.view(batch_size, num_tokens, -1)  # (Batch_Size, num_tokens, N)
    recon_logits_flat = recon_logits_flat.permute(0, 2, 1).reshape(-1, num_tokens)  # (N, num_tokens)
    data_tokens_flat = data_tokens.view(-1)  # (N,)

    # Create mask for non-air tokens
    mask = (data_tokens_flat != air_token_id)  # (N,)

    # Apply mask to outputs and targets
    recon_logits_flat = recon_logits_flat[mask]  # (N_non_air, num_tokens)
    data_tokens_flat = data_tokens_flat[mask]  # (N_non_air,)

    # Compute CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    recon_loss = criterion(recon_logits_flat, data_tokens_flat)

    # KL Divergence remains the same
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, KLD

# Function to interpolate and generate builds
def interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, epoch, num_interpolations=5, device='cpu'):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Load the two builds
        dataset = text2mcVAEDataset(
            file_paths=[build1_path, build2_path],
            block2tok=block2tok,
            fixed_size=fixed_size,
            augment=False  # Disable augmentation for consistent reconstructions
        )
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        data_tokens_list = []
        for data_tokens in data_loader:
            data_tokens = data_tokens.to(device)
            data_tokens_list.append(data_tokens)

        z_list = []
        for data_tokens in data_tokens_list:
            z, mu, logvar = encoder(data_tokens)
            z_list.append(z)

        # Generate reconstructions of the original builds
        for idx, (data_tokens, z) in enumerate(zip(data_tokens_list, z_list)):
            recon_logits = decoder(z)
            # Get predicted tokens
            recon_tokens = torch.argmax(recon_logits, dim=1)  # Shape: (1, D, H, W)
            recon_tokens_np = recon_tokens.cpu().numpy().squeeze(0)  # Shape: (D, H, W)

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
            recon_logits = decoder(z)
            # Get predicted tokens
            recon_tokens = torch.argmax(recon_logits, dim=1)  # Shape: (1, D, H, W)
            recon_tokens_np = recon_tokens.cpu().numpy().squeeze(0)  # Shape: (D, H, W)

            # Save the interpolated build as an HDF5 file
            save_path = os.path.join(save_dir, f'epoch_{epoch}_interp_{idx}.h5')
            with h5py.File(save_path, 'w') as h5f:
                h5f.create_dataset('build', data=recon_tokens_np, compression='gzip')

            print(f'Saved interpolated build at {save_path}')

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

# Create separate datasets with appropriate augmentation settings
train_dataset = text2mcVAEDataset(
    file_paths=train_file_paths,
    block2tok=block2tok,
    fixed_size=fixed_size,
    augment=True  # Enable augmentations for training
)

val_dataset = text2mcVAEDataset(
    file_paths=val_file_paths,
    block2tok=block2tok,
    fixed_size=fixed_size,
    augment=False  # Disable augmentations for validation
)

test_dataset = text2mcVAEDataset(
    file_paths=test_file_paths,
    block2tok=block2tok,
    fixed_size=fixed_size,
    augment=False  # Disable augmentations for testing
)

# Retrieve the air token ID from the training dataset
air_token_id = train_dataset.air_token
print(f"Air token ID: {air_token_id}")

# Get the number of tokens
max_token_id = train_dataset.max_token
num_tokens = max_token_id + 1  # Assuming tokens are 0-indexed

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model components
encoder = text2mcVAEEncoder(num_tokens, embedding_dim).to(device)
decoder = text2mcVAEDecoder(num_tokens, embedding_dim).to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

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

# Training loop
os.makedirs(save_dir, exist_ok=True)

for epoch in range(start_epoch, num_epochs + 1):
    encoder.train()
    decoder.train()
    average_reconstruction_loss = 0
    average_KL_divergence = 0

    for batch_idx, data_tokens in enumerate(train_loader):
        data_tokens = data_tokens.to(device)  # Tokens
        optimizer.zero_grad()

        z, mu, logvar = encoder(data_tokens)
        recon_logits = decoder(z)  # Outputs logits over tokens
        reconstruction_loss, KL_divergence = loss_function(recon_logits, data_tokens, mu, logvar, air_token_id=air_token_id)

        average_reconstruction_loss += reconstruction_loss.item()
        average_KL_divergence += KL_divergence.item()

        loss = reconstruction_loss + KL_divergence
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute accuracy every 30 batches
        if batch_idx % 30 == 0:
            with torch.no_grad():
                # Get predicted tokens
                recon_tokens = torch.argmax(recon_logits, dim=1)  # Shape: (Batch_Size, D, H, W)
                # Flatten the tokens
                recon_tokens_flat = recon_tokens.view(-1)
                data_tokens_flat = data_tokens.view(-1)
                # Create mask for non-air tokens
                mask = (data_tokens_flat != air_token_id)
                # Apply mask
                recon_tokens_non_air = recon_tokens_flat[mask]
                data_tokens_non_air = data_tokens_flat[mask]
                # Compute accuracy
                correct = (recon_tokens_non_air == data_tokens_non_air).sum().item()
                total = data_tokens_non_air.numel()
                accuracy = correct / total if total > 0 else 0.0
                # Compute other metrics if possible
                y_true = data_tokens_non_air.cpu().numpy()
                y_pred = recon_tokens_non_air.cpu().numpy()
                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            print(f'Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)] Reconstruction Error: {reconstruction_loss.item():.6f}, '
                  f'KL-divergence: {KL_divergence.item():.6f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
                  f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

    average_reconstruction_loss /= len(train_loader)
    average_KL_divergence /= len(train_loader)
    print(f'====> Epoch: {epoch} Average reconstruction loss: {average_reconstruction_loss:.8f}')
    print(f'====> Epoch: {epoch} Average KL-divergence: {average_KL_divergence:.8f}')

    # Validation
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for data_tokens in val_loader:
            data_tokens = data_tokens.to(device)

            z, mu, logvar = encoder(data_tokens)
            recon_logits = decoder(z)
            recon, kl = loss_function(recon_logits, data_tokens, mu, logvar, air_token_id=air_token_id)
            val_loss += (recon + kl).item()

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
        interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, epoch, num_interpolations=5, device=device)
    except Exception as e:
        print(f"Unable to generate interpolations for this epoch due to error: {e}")
