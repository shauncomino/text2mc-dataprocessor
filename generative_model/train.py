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
import torch.functional as F

batch_size = 4
num_epochs = 64
fixed_size = (128, 128, 128)
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

# Adjusted loss function with log-cosh loss
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

def interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, epoch, num_interpolations=5):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Load the two builds
        dataset = text2mcVAEDataset(file_paths=[build1_path, build2_path], tok2block=tok2block, block2embedding=block2embedding, fixed_size=fixed_size)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        data_list = []
        for data in data_loader:
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
            recon_tokens = embedding_to_tokens(recon_embedded, dataset.embeddings_matrix)
            # recon_tokens: (1, Depth, Height, Width)

            # Convert to numpy array
            recon_tokens_np = recon_tokens.cpu().numpy().squeeze(0)  # Shape: (Depth, Height, Width)

            # Save the build as an HDF5 file
            save_path = os.path.join(save_dir, f'epoch_{epoch}_interp_{idx}.h5')
            with h5py.File(save_path, 'w') as h5f:
                h5f.create_dataset('build', data=recon_tokens_np, compression='gzip')

            print(f'Saved interpolated build at {save_path}')


def embedding_to_tokens(embedded_data, embeddings_matrix):
    # embedded_data: (Batch_Size, Embedding_Dim, Depth, Height, Width)
    # embeddings_matrix: (Num_Tokens, Embedding_Dim)

    batch_size, embedding_dim, D, H, W = embedded_data.shape
    embedded_data = embedded_data.view(batch_size, embedding_dim, -1)  # Shape: (Batch_Size, Embedding_Dim, N)
    embedded_data = embedded_data.permute(0, 2, 1)  # Shape: (Batch_Size, N, Embedding_Dim)

    # Normalize embeddings
    embedded_data = F.normalize(embedded_data, dim=2)
    embeddings_matrix = embeddings_matrix.to(embedded_data.device)
    embeddings_matrix = F.normalize(embeddings_matrix, dim=1)

    # Compute cosine similarity
    similarity = torch.matmul(embedded_data, embeddings_matrix.T)  # Shape: (Batch_Size, N, Num_Tokens)

    # Find the most similar embedding (max cosine similarity)
    tokens = torch.argmax(similarity, dim=2)  # Shape: (Batch_Size, N)

    # Reshape tokens to (Batch_Size, Depth, Height, Width)
    tokens = tokens.view(batch_size, D, H, W)
    return tokens

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Device type
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
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

# Prepare the dataset
hdf5_filepaths = glob.glob(os.path.join(builds_folder_path, '*.h5'))
dataset = text2mcVAEDataset(file_paths=hdf5_filepaths, block2tok=block2tok, block2embedding=block2embedding, fixed_size=fixed_size)

print(f"Discovered {len(dataset)} builds, beginning training")

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
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
scaler = torch.amp.GradScaler()

start_epoch = 1
best_val_loss = float('inf')

# Load checkpoint if it exists
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
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
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(enabled=True, device_type='cuda'):
            z, mu, logvar = encoder(data)
            recon_batch = decoder(z)
            loss = loss_function(recon_batch, data, mu, logvar)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
        for data in val_loader:
            data = data.to(device)
            z, mu, logvar = encoder(data)
            recon_batch = decoder(z)
            loss = loss_function(recon_batch, data, mu, logvar)
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
        'scaler_state_dict': scaler.state_dict(),
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