import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from text2mcVAE import text2mcVAE
from text2mcVAEDataset import text2mcVAEDataset
from torch.utils.data import DataLoader

# Training functionality
def loss_function(recon_x, x, mu, logvar, mask):
    # Ensure mask is expanded to match the dimensions of recon_x
    # recon_x shape: [batch_size, C, D, H, W], we want mask to be [batch_size, 1, D, H, W]
    mask_expanded = mask.unsqueeze(1)  # Add a singleton dimension for channels

    # Ensure the mask is broadcastable across the channel dimension
    # recon_x might have more than one channel, replicate mask across channels
    mask_expanded = mask_expanded.expand_as(recon_x)  

    # Apply the mask
    recon_x_masked = recon_x * mask_expanded
    x_masked = x * mask_expanded

    # Calculate the Binary Cross-Entropy loss and KL Divergence
    BCE = nn.functional.binary_cross_entropy(recon_x_masked, x_masked, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = text2mcVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch, data_loader):
    model.train()
    total_loss = 0
    for batch_idx, (data, mask) in enumerate(data_loader):
        data, mask = data.to(device), mask.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)] Loss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {total_loss / len(data_loader.dataset):.4f}')

# Training function call
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
    # Ensure that the block name exists in the block2embedding dictionary before assigning
    if block_name in block2embedding:
        tok2embedding[token] = block2embedding[block_name]
    else:
        print(f"Warning: Block name '{block_name}' not found in embeddings. Skipping token '{token}'.")

hdf5_filepaths = [
    r'/mnt/d/processed_builds_compressed/rar_test5_Desert+Tavern+2.h5',
    r'/mnt/d/processed_builds_compressed/rar_test6_Desert_Tavern.h5',
    r'/mnt/d/processed_builds_compressed/zip_test_0_LargeSandDunes.h5'
]

dataset = text2mcVAEDataset(file_paths=hdf5_filepaths, tok2embedding=tok2embedding, block_ignore_list=[102])

data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

build, mask = next(iter(data_loader))
num_epochs = 1
for epoch in range(1, num_epochs + 1):
    train(epoch, data_loader)
# Convert tensors to numpy arrays
# build_np = build.numpy()
# mask_np = mask.numpy()

# # Find unique values and their counts in the build and mask arrays
# unique_build, counts_build = np.unique(build_np, return_counts=True)
# unique_mask, counts_mask = np.unique(mask_np, return_counts=True)

# # Output the results
# print("Build unique values and counts:", list(zip(unique_build, counts_build)))
# print("Mask unique values and counts:", list(zip(unique_mask, counts_mask)))


