import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from text2mcVAE import text2mcVAE
from text2mcVAEDataset import text2mcVAEDataset
from torch.utils.data import DataLoader

# Prediction functions
# Convert the predicted embeddings to their tokens
def embeddings_to_tokens(pred_embeddings, token_to_embedding):
    embeddings_array = np.array(list(token_to_embedding.values()))
    tokens_array = np.array(list(token_to_embedding.keys()))

    def nearest_token(embedding):
        distances = np.linalg.norm(embeddings_array - embedding, axis=1)
        return tokens_array[np.argmin(distances)]

    # Assuming pred_embeddings is a PyTorch tensor
    token_tensor = torch.zeros_like(pred_embeddings[..., 0], dtype=torch.long)
    for idx, embedding in np.ndenumerate(pred_embeddings):
        token_tensor[idx] = nearest_token(embedding.cpu().numpy())

    return token_tensor

# Convert token array to block name array
def tokens_to_block_names(pred_tokens, tok2block):
    # Vectorized conversion of tokens to block names
    vectorized_conversion = np.vectorize(lambda token: tok2block[token])
    block_names = vectorized_conversion(pred_tokens)
    return block_names

# Generate a full Minecraft build
def generate_minecraft_builds(model, input_data, token_to_embedding, tok2block):
    model.eval()
    with torch.no_grad():
        pred_embeddings = model(input_data)
        pred_tokens = embeddings_to_tokens(pred_embeddings, token_to_embedding)
        pred_block_names = tokens_to_block_names(pred_tokens.numpy(), tok2block)
    return pred_block_names


# Training functionality
def loss_function(recon_x, x, mu, logvar, mask):
    recon_x_masked = recon_x * mask.unsqueeze(1)
    x_masked = x * mask.unsqueeze(1)
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
training_df = pd.read_csv(r'projects_df_processed.csv')
tok2embedding = json.loads(r'tok2embedding.json')

hdf5_filepaths = list(training_df['FILEPATHS'])
dataset = text2mcVAEDataset(file_paths=hdf5_filepaths, token_to_embeddings=tok2embedding, block_ignore_list=["minecraft:air"])


data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch, data_loader)



