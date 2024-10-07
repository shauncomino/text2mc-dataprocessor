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
import math
from sklearn.metrics import precision_score, recall_score, f1_score  # Added for classification metrics

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
    build1_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_319_8281.h5'
    build2_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_225_5840.h5'
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

# Compute or load IDF weights
idf_weights_file = 'idf_weights.json'
if os.path.exists(idf_weights_file):
    print(f"Loading IDF weights from {idf_weights_file}")
    with open(idf_weights_file, 'r') as f:
        idf_weights = json.load(f)
        idf_weights = {int(k): float(v) for k, v in idf_weights.items()}
else:
    print("Computing IDF weights...")
    df = {}
    N = len(hdf5_filepaths)
    for file_path in hdf5_filepaths:
        with h5py.File(file_path, 'r') as file:
            build_folder_in_hdf5 = list(file.keys())[0]
            data = file[build_folder_in_hdf5][()]
        tokens_in_build = set(np.unique(data))
        for token in tokens_in_build:
            token = int(token)
            df[token] = df.get(token, 0) + 1
    # Compute IDF weights
    idf_weights = {}
    for token, df_t in df.items():
        idf_weights[token] = np.log((N + 1) / (df_t + 1)) + 1e-6  # Smooth IDF
    # Save IDF weights
    with open(idf_weights_file, 'w') as f:
        json.dump(idf_weights, f)
    print(f"Saved IDF weights to {idf_weights_file}")

# Create IDF weights tensor
max_token_id = train_dataset.max_token
idf_weights_tensor = torch.zeros(max_token_id + 1, dtype=torch.float32)
for token in range(max_token_id + 1):
    idf_weights_tensor[token] = idf_weights.get(token, 0.0)
idf_weights_tensor = idf_weights_tensor.to(device)

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

def loss_function(recon_x, x, mu, logvar, data_tokens, idf_weights_tensor, air_token_id, epsilon=1e-6):
    # recon_x and x: (Batch_Size, Embedding_Dim, D, H, W)
    # data_tokens: (Batch_Size, D, H, W)

    # Move Embedding_Dim to the last dimension
    recon_x = recon_x.permute(0, 2, 3, 4, 1)  # (Batch_Size, D, H, W, Embedding_Dim)
    x = x.permute(0, 2, 3, 4, 1)              # (Batch_Size, D, H, W, Embedding_Dim)

    # Flatten spatial dimensions
    batch_size, D, H, W, embedding_dim = recon_x.shape
    N = batch_size * D * H * W
    recon_x_flat = recon_x.reshape(N, embedding_dim)  # (N, Embedding_Dim)
    x_flat = x.reshape(N, embedding_dim)              # (N, Embedding_Dim)

    # Flatten data_tokens
    data_tokens_flat = data_tokens.reshape(-1)  # (N,)

    # Prepare labels for Cosine Embedding Loss
    y = torch.ones(x_flat.size(0), device=x_flat.device)  # (N,)

    # Compute Cosine Embedding Loss per voxel
    cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.0, reduction='none')
    loss_per_voxel = cosine_loss_fn(recon_x_flat, x_flat, y)  # (N,)

    # Compute TF per build within the batch
    tf_weights = torch.zeros_like(data_tokens, dtype=torch.float32)  # (Batch_Size, D, H, W)
    batch_size = data_tokens.size(0)

    for i in range(batch_size):
        tokens = data_tokens[i].view(-1)  # (D*H*W,)
        token_counts = torch.bincount(tokens, minlength=idf_weights_tensor.size(0))
        total_tokens = tokens.numel()
        tf = token_counts.float() / total_tokens  # (Num_Tokens,)
        # Map TF values back to the positions of the tokens
        tf_weights[i] = tf[tokens].view(D, H, W)

    # Flatten tf_weights
    tf_weights_flat = tf_weights.reshape(-1)  # (N,)

    # Get IDF weights per voxel
    idf_weights_per_voxel = idf_weights_tensor[data_tokens_flat]  # (N,)

    # Compute TF-IDF weights per voxel
    tf_idf_weights = tf_weights_flat * idf_weights_per_voxel  # (N,)

    # Multiply loss per voxel by TF-IDF weight
    weighted_loss = loss_per_voxel * tf_idf_weights  # (N,)

    # Create mask for non-air tokens
    mask = (data_tokens_flat != air_token_id)  # (N,)

    # Apply mask to weighted loss
    masked_weighted_loss = weighted_loss[mask]  # Only non-air tokens

    # Sum the masked weighted loss
    total_loss = masked_weighted_loss.sum()

    # Compute the number of non-air tokens
    num_non_air_tokens = mask.sum().float() + epsilon  # To avoid division by zero

    # Compute the mean loss
    recon_loss = total_loss / num_non_air_tokens

    # KL Divergence remains the same
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, KLD


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
    embedded_data_flat_norm = embedded_data_flat / (np.linalg.norm(embedded_data_flat, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    cosine_similarity = np.dot(embedded_data_flat_norm, embeddings_matrix_norm.T)  # Shape: (Batch_Size * N, Num_Tokens)

    # Find the token with the highest cosine similarity
    tokens_flat = np.argmax(cosine_similarity, axis=1)  # Shape: (Batch_Size * N,)

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
    average_reconstruction_loss = 0
    average_KL_divergence = 0

    for batch_idx, (data, data_tokens) in enumerate(train_loader):
        data = data.to(device)          # Embedded data
        data_tokens = data_tokens.to(device)  # Tokens
        optimizer.zero_grad()
        
        z, mu, logvar = encoder(data)
        recon_batch = decoder(z)
        reconstruction_loss, KL_divergence = loss_function(recon_batch, data, mu, logvar, data_tokens, idf_weights_tensor, air_token_id=air_token_id)

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
                # Convert reconstructions to tokens
                recon_tokens = embedding_to_tokens(recon_batch, train_dataset.embedding_matrix).to(device)
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
        for data, data_tokens in val_loader:
            data = data.to(device)
            data_tokens = data_tokens.to(device)
            
            z, mu, logvar = encoder(data)
            recon_batch = decoder(z)
            recon, kl = loss_function(recon_batch, data, mu, logvar, data_tokens, idf_weights_tensor, air_token_id=air_token_id)
            val_loss += (recon + kl)

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
