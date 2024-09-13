import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from text2mcVAE import text2mcVAE
from text2mcVAEDataset import text2mcVAEDataset
from embedder import text2mcVAEEmbedder
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Initialize the model components, optimizer, and gradient scaler

# Change the following line to the commented line proceeding it when using a capable machine to train
device_type = "cpu"
# device_type = "cuda" if torch.cuda.is_available() else "cpu"

tok2block = None
tok2block_file_path = r'../world2vec/tok2block.json'
with open(tok2block_file_path, 'r') as j:
    tok2block = json.loads(j.read())

embeddings = None

device = torch.device(device_type)
embedder = text2mcVAEEmbedder(emb_size=len(tok2block), emb_dimension=32).to(device)
encoder = text2mcVAEEncoder().to(device)
decoder = text2mcVAEDecoder().to(device)
optimizer = optim.Adam(list(embedder.parameters()) + list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
scaler = GradScaler()  # Initialize the gradient scaler for mixed precision

# Adjusted loss function remains the same
def loss_function(recon_x, x, mu, logvar):
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

def save_embeddings(embeddings, tok2block: dict[int, str]):
    embedding_dict = {}

    with open(os.path.join("embeddings", "embeddings.txt"), "w") as f:
        for tok, block_name in tok2block.items():
            e = " ".join(map(lambda x: str(x), embeddings[int(tok)]))
            embedding_dict[tok2block[str(tok)]] = torch.from_numpy(embeddings[int(tok)])
            f.write("%s %s\n" % (tok2block[str(tok)], e))
    np.save(os.path.join("embeddings", "embeddings.npy"), embeddings)

    # Create a copy of the embedding_dict with tensors converted to lists
    embedding_dict_copy = {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in embedding_dict.items()
    }

    # Write the modified copy to the JSON file
    with open(os.path.join("embeddings", "embeddings.json"), 'w') as f:
        json.dump(embedding_dict_copy, f)

def embed_block(block):
    print(torch.tensor(embeddings[tok2block[str(int(block))]]))
    return torch.tensor(embeddings[tok2block[str(int(block))]])


# Training function call
#block2embedding = None
#block2embedding_file_path = r'block2vec/output/block2vec/embeddings.json'
#with open(block2embedding_file_path, 'r') as j:
    #block2embedding = json.loads(j.read())

# Create a new dictionary mapping tokens directly to embeddings
#tok2embedding = {}

#for token, block_name in tok2block.items():
    #if block_name in block2embedding:
        #tok2embedding[token] = block2embedding[block_name]
    #else:
        #print(f"Warning: Block name '{block_name}' not found in embeddings. Skipping token '{token}'.")

hdf5_filepaths = [
    'batch_400_10391.h5',
]

dataset = text2mcVAEDataset(file_paths=hdf5_filepaths, tok2block=tok2block)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

build, target, context = next(iter(data_loader))
num_epochs = 1

# Training loop accessible at the end of the file
for epoch in range(1, num_epochs + 1):
    embedder.train()
    encoder.train()
    decoder.train()
    total_loss = 0
    
    for batch_idx, data in enumerate(data_loader):
        build_data, targets, contexts = data
        build_data = build_data.to(device)
        targets = [torch.tensor(target, dtype=torch.int64).to(device) for target in targets if len(target) > 0]
        contexts = [torch.tensor(context, dtype=torch.int64).to(device) for context in contexts if len(context) > 0]
        optimizer.zero_grad()

        # Mixed precision context
        with autocast(device_type=device_type):
            # Embed the data
            embedding_loss = 0
            index = 0
            for target_block in targets[0]:
                embedding_loss += embedder(target_block, contexts[0][index])
                index += 1
            embedding_loss /= index + 1
            embeddings_array = embedder.target_embeddings.weight.cpu().data.numpy()
            save_embeddings(embeddings_array, tok2block)

            with open(os.path.join("embeddings", "embeddings.json"), 'r') as f:
                embeddings = json.loads(f.read())
            
            np.vectorize(embed_block)(build_data)

            #embedded_data = build_data[:, :, :, :, None].expand(-1, -1, -1, -1, 32).clone()
            #for x in range(0, 256):
                #for y in range(0, 256):
                    #for z in range(0, 256):
                        #embedded_data[0][x][y][z] = torch.tensor(embeddings[int(embedded_data[0][x][y][z][0])])
                        #print(x, y, z)

            # Encode the data to get latent representation, mean, and log variance
            z, mu, logvar = encoder(build_data)

            # Decode the latent variable to reconstruct the input
            recon_batch = decoder(z)

            # Compute the loss
            loss = embedding_loss + loss_function(recon_batch, build_data, mu, logvar)

        # Scale loss and perform backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)] Loss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {total_loss / len(data_loader.dataset):.4f}')
