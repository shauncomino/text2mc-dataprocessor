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
import torch.functional as F

batch_size = 2
num_epochs = 64
fixed_size = (64, 64, 64)
embedding_dim = 32

# Paths and configurations
tok2block_file_path = r'/home/shaun/projects/text2mc-dataprocessor/world2vec/tok2block.json'
build1_path = r'/home/shaun/projects/text2mc-dataprocessor/test_builds/batch_157_4066_4.h5'
build2_path = r'/home/shaun/projects/text2mc-dataprocessor/test_builds/batch_158_4104.h5'
save_dir = r'/home/shaun/projects/text2mc-dataprocessor/test_builds'
block2embedding_file_path = r'/home/shaun/projects/text2mc-dataprocessor/block2vec/output/block2vec/embeddings.json'
builds_folder_path = r'/home/shaun/projects/text2mc-dataprocessor/test_builds'


# Load mappings
with open(tok2block_file_path, 'r') as f:
    tok2block = json.load(f)
    tok2block = {int(k): v for k, v in tok2block.items()}  # Ensure keys are integers

block2tok = {v: k for k, v in tok2block.items()}

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
        dataset = text2mcVAEDataset(file_paths=[build1_path, build2_path], block2tok=block2tok, block2embedding=block2embedding, fixed_size=fixed_size)
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
            recon_tokens = embedding_to_tokens(recon_embedded, dataset.embedding_matrix)
            # recon_tokens: (1, Depth, Height, Width)

            # Convert to numpy array
            recon_tokens_np = recon_tokens.cpu().numpy().squeeze(0)  # Shape: (Depth, Height, Width)

            # Save the build as an HDF5 file
            save_path = os.path.join(save_dir, f'epoch_{epoch}_interp_{idx}.h5')
            with h5py.File(save_path, 'w') as h5f:
                h5f.create_dataset('build', data=recon_tokens_np, compression='gzip')

            print(f'Saved interpolated build at {save_path}')

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



# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Device type
# device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cpu'
device = torch.device(device_type)
print("Using device:", device)



with open(block2embedding_file_path, 'r') as f:
    block2embedding = json.load(f)
    block2embedding = {k: np.array(v, dtype=np.float32) for k, v in block2embedding.items()}

# Initialize the model components
encoder = text2mcVAEEncoder().to(device)
decoder = text2mcVAEDecoder().to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
scaler = torch.amp.GradScaler()

interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, 0, num_interpolations=5)
