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
from torch.utils.data import DataLoader
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

# Adjusted loss function remains the same
def loss_function(recon_x, x, mu, logvar, target, context):
    # If recon_x has more channels, slice to match x's channels
    recon_x = recon_x[:, :x.size(1), :, :, :]

    # Adjust the mask to match the shape of recon_x
    #mask_expanded = mask.unsqueeze(1)  # Add a singleton dimension for channels
    #mask_expanded = mask_expanded.expand_as(recon_x)  # Expand to match recon_x's channel size

    # Apply the mask
    #recon_x_masked = recon_x * mask_expanded
    #x_masked = x * mask_expanded

    # Calculate the Binary Cross-Entropy loss with logits
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')

    # Calculate KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD




# Training function call
tok2block = None
#block2embedding = None
#block2embedding_file_path = r'block2vec/output/block2vec/embeddings.json'
tok2block_file_path = r'world2vec/tok2block.json'
#with open(block2embedding_file_path, 'r') as j:
    #block2embedding = json.loads(j.read())

with open(tok2block_file_path, 'r') as j:
    tok2block = json.loads(j.read())

# Create a new dictionary mapping tokens directly to embeddings
#tok2embedding = {}

#for token, block_name in tok2block.items():
    #if block_name in block2embedding:
        #tok2embedding[token] = block2embedding[block_name]
    #else:
        #print(f"Warning: Block name '{block_name}' not found in embeddings. Skipping token '{token}'.")

hdf5_filepaths = [
    r'/mnt/d/processed_builds_compressed/rar_test5_Desert+Tavern+2.h5',
    # r'/mnt/d/processed_builds_compressed/rar_test6_Desert_Tavern.h5',
    # r'/mnt/d/processed_builds_compressed/zip_test_0_LargeSandDunes.h5'
]

dataset = text2mcVAEDataset(file_paths=hdf5_filepaths, tok2block=tok2block)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

build, target, context = next(iter(data_loader))
num_epochs = 1

# Training loop accessible at the end of the file
for epoch in range(1, num_epochs + 1):
    encoder.train()
    decoder.train()
    total_loss = 0
    
    for batch_idx, data, target, context in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        context = context.to(device)
        optimizer.zero_grad()

        # Mixed precision context
        with autocast(device_type=device_type):
            # Encode the data to get latent representation, mean, and log variance
            z, mu, logvar = encoder(data, target, context)

            # Decode the latent variable to reconstruct the input
            recon_batch = decoder(z)

            # Compute the loss
            loss = loss_function(recon_batch, data, mu, logvar, target, context)

        # Scale loss and perform backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)] Loss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {total_loss / len(data_loader.dataset):.4f}')
