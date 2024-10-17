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
    checkpoint_path = r'/lustre/fs1/home/scomino/axial_attention_training/checkpoint.pth'
    tok2block_file_path = r'/lustre/fs1/home/scomino/axial_attention/text2mc-dataprocessor/world2vec/tok2block.json'
    builds_folder_path = r'/lustre/fs1/groups/jaedo/processed_builds'
    build1_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_319_8281.h5'
    build2_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_225_5840.h5'
    save_dir = r'/lustre/fs1/home/scomino/axial_attention_training/interpolations'
    best_model_path = r'/lustre/fs1/home/scomino/axial_attention_training/best_model.pth'
    block2embedding_file_path = r'/lustre/fs1/home/scomino/axial_attention/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'
    
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

    # Ensure embeddings_pred and x have matching spatial dimensions
    assert embeddings_pred.shape == x.shape, f"Shape mismatch: embeddings_pred {embeddings_pred.shape}, x {x.shape}"

    # Move Embedding_Dim to the last dimension
    embeddings_pred = embeddings_pred.permute(0, 2, 3, 4, 1).contiguous()
    x = x.permute(0, 2, 3, 4, 1).contiguous()

    # Flatten spatial dimensions
    batch_size, D, H, W, embedding_dim = embeddings_pred.shape
    N = batch_size * D * H * W

    embeddings_pred_flat = embeddings_pred.view(N, embedding_dim)
    x_flat = x.view(N, embedding_dim)

    # Flatten data_tokens
    data_tokens_flat = data_tokens.view(-1)

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
    block_air_pred_probs = block_air_pred.squeeze(1)
    bce_loss = bce_loss_fn(block_air_pred_probs, block_air_labels)

    # Compute KL Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine losses
    total_loss = recon_loss + bce_loss + KLD

    return total_loss, recon_loss, bce_loss, KLD

# Function to convert embeddings back to tokens
def embedding_to_tokens(embeddings_pred, embedding_matrix):
    # embeddings_pred: PyTorch tensor of shape (Batch_Size, Embedding_Dim, D, H, W)
    # embedding_matrix: NumPy array of shape (Num_Tokens, Embedding_Dim)

    # Move Embedding_Dim to last dimension
    embeddings_pred = embeddings_pred.permute(0, 2, 3, 4, 1).contiguous()  # Shape: (Batch_Size, D, H, W, Embedding_Dim)
    batch_size, D, H, W, embedding_dim = embeddings_pred.shape
    N = batch_size * D * H * W

    # Flatten embeddings
    embeddings_pred_flat = embeddings_pred.view(-1, embedding_dim).cpu().numpy()  # Shape: (N, Embedding_Dim)

    # Normalize embeddings_pred_flat
    embeddings_pred_flat_norm = embeddings_pred_flat / (np.linalg.norm(embeddings_pred_flat, axis=1, keepdims=True) + 1e-8)

    # Normalize embedding_matrix
    embedding_matrix_norm = embedding_matrix / (np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    cosine_similarity = np.dot(embeddings_pred_flat_norm, embedding_matrix_norm.T)  # Shape: (N, Num_Tokens)

    # Find the token with the highest cosine similarity
    tokens_flat = np.argmax(cosine_similarity, axis=1)  # Shape: (N,)

    # Reshape tokens back to (Batch_Size, D, H, W)
    tokens = tokens_flat.reshape(batch_size, D, H, W)
    tokens = torch.from_numpy(tokens).long()  # Convert to torch tensor

    return tokens

# Function to interpolate and generate builds
def interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, epoch, num_interpolations=20):
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
        data_tokens_list = []
        for data, data_tokens in data_loader:
            data = data.to(device)
            data_tokens = data_tokens.to(device)
            data_list.append(data)
            data_tokens_list.append(data_tokens)

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
            embeddings_pred, block_air_pred = decoder(z)
            # Convert embeddings back to tokens
            recon_tokens = embedding_to_tokens(embeddings_pred, train_dataset.embedding_matrix).to(device)
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
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_batches = 0

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
        
        # Compute classification metrics
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

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_batches += 1

        # Logging
        if batch_idx % 30 == 0:
            print(f'Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)}] '
                  f'Recon Loss: {recon_loss.item():.6f}, KL Divergence: {KLD.item():.6f}, '
                  f'BCE Loss: {bce_loss.item():.6f}, F1 Score: {f1:.4f}')

    # Compute average losses and metrics over the epoch
    average_reconstruction_loss /= num_batches
    average_bce_loss /= num_batches
    average_KL_divergence /= num_batches
    average_accuracy = total_accuracy / num_batches
    average_precision = total_precision / num_batches
    average_recall = total_recall / num_batches
    average_f1 = total_f1 / num_batches

    print(f'====> Epoch: {epoch} Average Reconstruction Loss: {average_reconstruction_loss:.8f}')
    print(f'====> Epoch: {epoch} Average KL-Divergence: {average_KL_divergence:.8f}')
    print(f'====> Epoch: {epoch} Average BCE Loss: {average_bce_loss:.8f}')
    print(f'====> Epoch: {epoch} Average Block-Air Accuracy: {average_accuracy:.4f}, '
          f'Precision: {average_precision:.4f}, Recall: {average_recall:.4f}, F1 Score: {average_f1:.4f}')

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
            z, mu, logvar = encoder(data)
            embeddings_pred, block_air_pred = decoder(z)
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
    print(f'====> Validation Block-Air Accuracy: {avg_val_accuracy:.4f}, '
          f'Precision: {avg_val_precision:.4f}, Recall: {avg_val_recall:.4f}, F1 Score: {avg_val_f1:.4f}')

    # Save the best model
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
