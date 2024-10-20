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
from sklearn.metrics import precision_score, recall_score, f1_score  # Added for classification metrics
from collections import Counter

batch_size = 6
num_epochs = 128
fixed_size = (64, 64, 64)
embedding_dim = 32
on_arcc = True

if on_arcc:
    # Paths and configurations
    checkpoint_path = r'/lustre/fs1/home/scomino/weighted_training/checkpoint.pth'
    tok2block_file_path = r'/lustre/fs1/home/scomino/weighted/text2mc-dataprocessor/world2vec/tok2block.json'
    builds_folder_path = r'/lustre/fs1/groups/jaedo/processed_builds'
    build1_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_319_8281.h5'
    build2_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_225_5840.h5'
    save_dir = r'/lustre/fs1/home/scomino/weighted_training/interpolations'
    best_model_path = r'/lustre/fs1/home/scomino/weighted_training/best_model.pth'
    block2embedding_file_path = r'/lustre/fs1/home/scomino/weighted/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'
    
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
    # Load embeddings
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

# Get the maximum token ID
max_token_id = train_dataset.max_token

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model components
encoder = text2mcVAEEncoder().to(device)
decoder = text2mcVAEDecoder().to(device)
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

def loss_function(
    embeddings_pred, block_air_pred, x, mu, logvar,
    data_tokens, air_token_id, epsilon=1e-8
):
    # embeddings_pred and x: (Batch_Size, Embedding_Dim, D, H, W)
    # block_air_pred: (Batch_Size, 1, D, H, W)
    # data_tokens: (Batch_Size, D, H, W)

    # Move Embedding_Dim to the last dimension
    embeddings_pred = embeddings_pred.permute(0, 2, 3, 4, 1)  # Shape: (B, D, H, W, E)
    x = x.permute(0, 2, 3, 4, 1)                              # Shape: (B, D, H, W, E)

    batch_size, D, H, W, embedding_dim = embeddings_pred.shape
    N_per_sample = D * H * W
    N = batch_size * N_per_sample

    # Flatten spatial dimensions
    embeddings_pred_flat = embeddings_pred.reshape(N, embedding_dim)  # Shape: (N, E)
    x_flat = x.reshape(N, embedding_dim)                              # Shape: (N, E)
    data_tokens_flat = data_tokens.view(batch_size, -1)               # Shape: (B, N_per_sample)

    # Compute Cosine Embedding Loss per voxel without reduction
    # Prepare labels for Cosine Embedding Loss
    y = torch.ones(N, device=x_flat.device)  # Shape: (N,)

    # Normalize embeddings for cosine similarity computation
    embeddings_pred_norm = F.normalize(embeddings_pred_flat, p=2, dim=1)
    x_norm = F.normalize(x_flat, p=2, dim=1)

    # Compute cosine similarity
    cosine_similarity = torch.sum(embeddings_pred_norm * x_norm, dim=1)  # Shape: (N,)

    # Compute per-voxel loss
    loss_per_voxel = 1 - cosine_similarity  # Shape: (N,)

    # Reshape loss_per_voxel to (batch_size, N_per_sample)
    loss_per_voxel_flat = loss_per_voxel.view(batch_size, N_per_sample)

    # Prepare block-air predictions and labels
    block_air_pred_probs = block_air_pred.squeeze(1)  # Shape: (B, D, H, W)
    block_air_pred_probs_flat = block_air_pred_probs.view(batch_size, -1)  # Shape: (B, N_per_sample)
    block_air_labels = (data_tokens != air_token_id).float()  # Shape: (B, D, H, W)
    block_air_labels_flat = block_air_labels.view(batch_size, -1)  # Shape: (B, N_per_sample)

    total_weighted_loss = 0.0
    total_token_weights = 0.0
    bce_loss_total = 0.0
    bce_weights_total = 0.0

    # Determine the maximum token ID in the batch
    max_token_id = data_tokens.max().item()

    for i in range(batch_size):
        tokens_i = data_tokens_flat[i]  # Shape: (N_per_sample,)
        loss_per_voxel_i = loss_per_voxel_flat[i]  # Shape: (N_per_sample,)
        block_air_pred_i = block_air_pred_probs_flat[i]  # Shape: (N_per_sample,)
        block_air_label_i = block_air_labels_flat[i]  # Shape: (N_per_sample,)

        # Exclude air tokens
        mask_i = (tokens_i != air_token_id)
        tokens_i_non_air = tokens_i[mask_i]  # Shape: (num_non_air_voxels_in_sample,)
        loss_per_voxel_i_non_air = loss_per_voxel_i[mask_i]  # Shape: (num_non_air_voxels_in_sample,)
        block_air_pred_i_non_air = block_air_pred_i[mask_i]
        block_air_label_i_non_air = block_air_label_i[mask_i]

        # Compute counts and probabilities for the current sample
        unique_tokens, counts = torch.unique(tokens_i_non_air, return_counts=True)
        total_tokens = tokens_i_non_air.numel()
        token_probs = counts.float() / total_tokens  # Probabilities per token in this sample

        # Create a tensor of size (max_token_id + 1,) filled with zeros
        probs_per_sample = torch.zeros(max_token_id + 1, device=data_tokens.device)
        probs_per_sample[unique_tokens] = token_probs

        # Get probabilities for tokens_i_non_air
        token_probs_i = probs_per_sample[tokens_i_non_air]  # Shape: (num_non_air_voxels_in_sample,)

        # Compute weights: w = 1 - probability_of_token
        token_weights_i = 1.0 - token_probs_i  # Shape: (num_non_air_voxels_in_sample,)

        # Compute weighted loss for reconstruction
        weighted_loss_per_voxel_i = loss_per_voxel_i_non_air * token_weights_i
        total_weighted_loss += weighted_loss_per_voxel_i.sum()
        total_token_weights += token_weights_i.sum()

        # For BCE loss:
        # Include air tokens but don't weight them
        bce_weights_i = torch.ones_like(block_air_label_i)
        bce_weights_i[mask_i] = token_weights_i

        # Compute BCE loss per element
        bce_loss_per_element_i = F.binary_cross_entropy(
            block_air_pred_i, block_air_label_i, reduction='none'
        )
        # Apply weights
        weighted_bce_loss_per_element_i = bce_loss_per_element_i * bce_weights_i

        bce_loss_total += weighted_bce_loss_per_element_i.sum()
        bce_weights_total += bce_weights_i.sum()

    # Compute mean reconstruction loss
    recon_loss = total_weighted_loss / (total_token_weights + epsilon)
    # Compute mean BCE loss
    bce_loss = bce_loss_total / (bce_weights_total + epsilon)

    # Compute KL Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine losses
    total_loss = recon_loss + bce_loss + KLD

    return total_loss, recon_loss, bce_loss, KLD





# Function to convert embeddings back to tokens
def embedding_to_tokens(embedded_data, embeddings_matrix):
    # embedded_data: PyTorch tensor of shape (Batch_Size, Embedding_Dim, D, H, W)
    # embeddings_matrix: NumPy array or PyTorch tensor of shape (Num_Tokens, Embedding_Dim)

    batch_size, embedding_dim, D, H, W = embedded_data.shape

    # Convert embedded_data to NumPy array
    embedded_data_np = embedded_data.detach().cpu().numpy()

    # Ensure embeddings_matrix is a NumPy array
    if isinstance(embeddings_matrix, torch.Tensor):
        embeddings_matrix_np = embeddings_matrix.detach().cpu().numpy()
    else:
        embeddings_matrix_np = embeddings_matrix

    embeddings_matrix_norm = embeddings_matrix_np

    # Flatten the embedded data
    N = D * H * W
    embedded_data_flat = embedded_data_np.reshape(batch_size, embedding_dim, N)
    embedded_data_flat = embedded_data_flat.transpose(0, 2, 1)  # Shape: (Batch_Size, N, Embedding_Dim)
    embedded_data_flat = embedded_data_flat.reshape(-1, embedding_dim)  # Shape: (Batch_Size * N, Embedding_Dim)

    # Normalize embedded_data_flat
    embedded_data_flat_norm = embedded_data_flat

    # Compute cosine similarity
    cosine_similarity = np.dot(embedded_data_flat_norm, embeddings_matrix_norm.T)  # Shape: (Batch_Size * N, Num_Tokens)

    # Find the token with the highest cosine similarity
    tokens_flat = np.argmax(cosine_similarity, axis=1)  # Shape: (Batch_Size * N,)

    # Reshape tokens back to (Batch_Size, Depth, Height, Width)
    tokens = tokens_flat.reshape(batch_size, D, H, W)
    tokens = torch.from_numpy(tokens).long()  # Convert to torch tensor

    return tokens

# Function to interpolate and generate builds
def interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, epoch, num_interpolations=60):
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
        for data, _ in data_loader:
            data = data.to(device)
            data_list.append(data)

        z_list = []
        for data in data_list:
            z, mu, logvar = encoder(data)
            z_list.append(z)

        # Generate reconstructions of the original builds
        for idx, (data, z) in enumerate(zip(data_list, z_list)):
            embeddings_pred, block_air_pred = decoder(z)
            # Convert embeddings back to tokens
            recon_tokens = embedding_to_tokens(embeddings_pred, dataset.embedding_matrix).to(device)
            # Apply block-air mask
            block_air_pred_labels = (block_air_pred.squeeze(1) >= 0.5).long()
            air_mask = (block_air_pred_labels == 0)
            # Assign air_token_id to air voxels
            recon_tokens[air_mask] = air_token_id
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
            embeddings_pred, block_air_pred = decoder(z)
            # Convert embeddings back to tokens
            recon_tokens = embedding_to_tokens(embeddings_pred, dataset.embedding_matrix).to(device)
            # Apply block-air mask
            block_air_pred_labels = (block_air_pred.squeeze(1) >= 0.5).long()
            air_mask = (block_air_pred_labels == 0)
            # Assign air_token_id to air voxels
            recon_tokens[air_mask] = air_token_id
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
        
        # Compute classification metrics every 30 batches
        if batch_idx % 30 == 0:
            with torch.no_grad():
                # Prepare block-air predictions and labels
                block_air_pred_probs = block_air_pred.squeeze(1)  # Shape: (Batch_Size, D, H, W)
                block_air_pred_labels = (block_air_pred_probs >= 0.5).long()
                block_air_labels = (data_tokens != air_token_id).long()

                # Flatten tensors
                block_air_pred_flat = block_air_pred_labels.view(-1).cpu()
                block_air_labels_flat = block_air_labels.view(-1).cpu()

                # Compute classification metrics
                accuracy = (block_air_pred_flat == block_air_labels_flat).sum().item() / block_air_labels_flat.numel()
                precision = precision_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)
                recall = recall_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)
                f1 = f1_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)

            print(f'Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)] Reconstruction Error: {recon_loss.item():.6f}, '
                  f'KL-Divergence: {KLD.item():.6f}, BCE Loss: {bce_loss.item():.6f}, '
                  f'Block-Air Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
                  f'Recall: {recall:.4f}, F1-Score: {f1:.4f}')

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
    val_recon_loss = 0
    val_bce_loss = 0
    val_KLD = 0
    val_accuracy = 0
    val_precision = 0
    val_recall = 0
    val_f1 = 0
    val_batches = 0

    with torch.no_grad():
        for data, data_tokens in val_loader:
            data = data.to(device)
            data_tokens = data_tokens.to(device)
            
            # Forward pass
            z, mu, logvar = encoder(data)
            embeddings_pred, block_air_pred = decoder(z)
            
            # Compute losses
            total_loss, recon_loss, bce_loss, KLD = loss_function(
                embeddings_pred, block_air_pred, data, mu, logvar, data_tokens, air_token_id=air_token_id
            )
            val_loss += total_loss.item()
            val_recon_loss += recon_loss.item()
            val_bce_loss += bce_loss.item()
            val_KLD += KLD.item()
            val_batches += 1

            # Prepare block-air predictions and labels
            block_air_pred_probs = block_air_pred.squeeze(1)
            block_air_pred_labels = (block_air_pred_probs >= 0.5).long()
            block_air_labels = (data_tokens != air_token_id).long()

            # Flatten tensors
            block_air_pred_flat = block_air_pred_labels.view(-1).cpu()
            block_air_labels_flat = block_air_labels.view(-1).cpu()

            # Compute classification metrics
            accuracy = (block_air_pred_flat == block_air_labels_flat).sum().item() / block_air_labels_flat.numel()
            precision = precision_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)
            recall = recall_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)
            f1 = f1_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)

            val_accuracy += accuracy
            val_precision += precision
            val_recall += recall
            val_f1 += f1

    # Compute average validation losses and metrics
    avg_val_loss = val_loss / val_batches
    avg_val_recon_loss = val_recon_loss / val_batches
    avg_val_bce_loss = val_bce_loss / val_batches
    avg_val_KLD = val_KLD / val_batches
    avg_val_accuracy = val_accuracy / val_batches
    avg_val_precision = val_precision / val_batches
    avg_val_recall = val_recall / val_batches
    avg_val_f1 = val_f1 / val_batches

    print(f'====> Epoch: {epoch} Validation Loss: {avg_val_loss:.4f}')
    print(f'====> Validation Reconstruction Loss: {avg_val_recon_loss:.6f}')
    print(f'====> Validation KL-Divergence: {avg_val_KLD:.6f}')
    print(f'====> Validation BCE Loss: {avg_val_bce_loss:.6f}')
    print(f'====> Validation Block-Air Accuracy: {avg_val_accuracy:.4f}, Precision: {avg_val_precision:.4f}, '
          f'Recall: {avg_val_recall:.4f}, F1-Score: {avg_val_f1:.4f}')

    # Save the best model based on validation loss
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
        interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, epoch)
    except Exception as e:
        print(f"Unable to generate interpolations for this epoch due to error: {e}")
