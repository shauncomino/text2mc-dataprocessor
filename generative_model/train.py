import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from text2mcVAE import text2mcVAE
from text2mcVAEDataset import text2mcVAEDataset
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

# Initialize the model components, optimizer, and gradient scaler

# Change the following line to the commented line proceeding it when using a capable machine to train
device_type = "cpu"
# device_type = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device(device_type)
encoder = text2mcVAEEncoder().to(device)
decoder = text2mcVAEDecoder().to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
scaler = GradScaler()  # Initialize the gradient scaler for mixed precision

# Adjusted loss function with log-cosh loss
def log_cosh_loss(x, y, a=1.0):
    diff = a * (x - y)
    return torch.mean((1.0 / a) * torch.log(torch.cosh(diff)))

def loss_function(recon_x, x, mu, logvar, mask, a=10.0):
    # If recon_x has more channels, slice to match x's channels
    recon_x = recon_x[:, :x.size(1), :, :, :]

    # Adjust the mask to match the shape of recon_x
    mask_expanded = mask.unsqueeze(1)  # Add a singleton dimension for channels
    mask_expanded = mask_expanded.expand_as(recon_x)  # Expand to match recon_x's channel size

    # Apply the mask
    recon_x_masked = recon_x * mask_expanded
    x_masked = x * mask_expanded

    # Use log-cosh error instead of MSE
    log_cosh = log_cosh_loss(recon_x_masked, x_masked, a)

    # Calculate KL Divergence
    batch_size = x.size(0)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    return log_cosh + KLD

# Load tok2embedding
tok2block = None
block2embedding = None
block2embedding_file_path = r'block2vec/output/block2vec/embeddings.json'
tok2block_file_path = r'world2vec/tok2block.json'
with open(block2embedding_file_path, 'r') as j:
    block2embedding = json.loads(j.read())

with open(tok2block_file_path, 'r') as j:
    tok2block = json.loads(j.read())

# Create a new dictionary mapping tokens directly to embeddings
tok2embedding = {}

for token, block_name in tok2block.items():
    if block_name in block2embedding:
        tok2embedding[token] = block2embedding[block_name]
    else:
        print(f"Warning: Block name '{block_name}' not found in embeddings. Skipping token '{token}'.")

hdf5_filepaths = [
    r'/mnt/d/processed_builds_compressed/rar_test5_Desert+Tavern+2.h5',
    # r'/mnt/d/processed_builds_compressed/rar_test6_Desert_Tavern.h5',
    # r'/mnt/d/processed_builds_compressed/zip_test_0_LargeSandDunes.h5'
]

dataset = text2mcVAEDataset(file_paths=hdf5_filepaths, tok2embedding=tok2embedding, block_ignore_list=[102])

# Split the dataset into training, validation, and test sets
dataset_size = len(dataset)
validation_split = 0.2
test_split = 0.1
train_size = int((1 - validation_split - test_split) * dataset_size)
val_size = int(validation_split * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
batch_size = 1  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 1  # Adjust as needed

# Training loop
best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch_idx, (data, mask) in enumerate(train_loader):
        data, mask = data.to(device), mask.to(device)
        optimizer.zero_grad()

        # Mixed precision context
        with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            # Encode the data to get latent representation, mean, and log variance
            z, mu, logvar = encoder(data)

            # Decode the latent variable to reconstruct the input
            recon_batch = decoder(z)

            # Compute the loss
            loss = loss_function(recon_batch, data, mu, logvar, mask)

        # Scale loss and perform backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}')
    avg_train_loss = total_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average training loss: {avg_train_loss:.4f}')

    # Validation
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for data, mask in val_loader:
            data, mask = data.to(device), mask.to(device)
            z, mu, logvar = encoder(data)
            recon_batch = decoder(z)
            loss = loss_function(recon_batch, data, mu, logvar, mask)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader.dataset)
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
        }, 'best_model.pth')
        print(f'Saved new best model at epoch {epoch} with validation loss {avg_val_loss:.4f}')
