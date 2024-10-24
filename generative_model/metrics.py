import json
import torch
import torch.nn as nn
import os
import sys
import h5py
import pandas as pd
from metricsVAEDataset import metircsVAEDataset
from predictor import text2mcPredictor
from torch.utils.data import DataLoader
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from sklearn.metrics import precision_score, recall_score, f1_score

air_token_id = 102

def loss_function(embeddings_pred, block_air_pred, x, mu, logvar, data_tokens, air_token_id, epsilon=1e-8):
    # embeddings_pred and x: (Batch_Size, Embedding_Dim, D, H, W)
    # block_air_pred: (Batch_Size, 1, D, H, W)
    # data_tokens: (Batch_Size, D, H, W)

    # Move Embedding_Dim to the last dimension
    embeddings_pred = embeddings_pred.permute(0, 2, 3, 4, 1)  # Shape: (Batch_Size, D, H, W, Embedding_Dim)
    x = x.permute(0, 2, 3, 4, 1)                              # Shape: (Batch_Size, D, H, W, Embedding_Dim)

    # Flatten spatial dimensions
    batch_size, D, H, W, embedding_dim = embeddings_pred.shape
    N = batch_size * D * H * W
    embeddings_pred_flat = embeddings_pred.reshape(N, embedding_dim)  # Shape: (N, Embedding_Dim)
    x_flat = x.reshape(N, embedding_dim)                              # Shape: (N, Embedding_Dim)

    # Flatten data_tokens
    data_tokens_flat = data_tokens.reshape(-1)  # Shape: (N,)

    print(x_flat.device)
    # Prepare labels for Cosine Embedding Loss
    y = torch.ones(N, device=x_flat.device)  # Shape: (N,)

    # Compute Cosine Embedding Loss per voxel without reduction
    cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.0, reduction='none')
    loss_per_voxel = cosine_loss_fn(embeddings_pred_flat, x_flat, y)  # Shape: (N,)

    # Mask out the error tensor to include only the errors of the non-air blocks
    mask = (data_tokens_flat != air_token_id)  # Shape: (N,)
    loss_per_voxel = loss_per_voxel[mask]      # Shape: (num_non_air_voxels,)

    # Compute mean over non-air blocks
    num_non_air_voxels = loss_per_voxel.numel() + epsilon  # To prevent division by zero
    recon_loss = loss_per_voxel.sum() / num_non_air_voxels

    # Prepare ground truth labels for block vs. air
    block_air_labels = (data_tokens != air_token_id).float()  # Shape: (Batch_Size, D, H, W)

    # Compute binary cross-entropy loss for block vs. air prediction over all voxels
    bce_loss_fn = nn.BCELoss()
    block_air_pred = block_air_pred.squeeze(1)  # Shape: (Batch_Size, D, H, W)
    bce_loss = bce_loss_fn(block_air_pred, block_air_labels)

    # Compute KL Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine losses
    total_loss = recon_loss + bce_loss + KLD

    return total_loss, recon_loss, bce_loss, KLD

# Step 1: Read the CSV file and make a copy of it
csv_file_path = "/lustre/fs1/groups/jaedo/generative_model/batch_16.csv"
df = pd.read_csv(csv_file_path)
df_copy = df.copy()

# Handle NaN values and ensure all values are lists
df_copy['PROCESSED_PATHS'] = df_copy['PROCESSED_PATHS'].apply(lambda x: [] if pd.isna(x) else x.strip("[]").replace("'", "").split(','))

# Extract the paths from the 'PROCESSED_PATHS' column
paths_list = df_copy['PROCESSED_PATHS']

# Flatten the list of lists into a single list and remove empty lists
all_paths = [path.strip() for sublist in paths_list for path in sublist if path.strip()]

# Replace .schem with .h5
all_paths = [path.replace('.schem', '.h5') for path in all_paths]

# Filter out non-existent files
valid_paths = [path for path in all_paths if os.path.exists(path)]

# Print the valid paths for debugging
print("Valid HDF5 paths:")
for path in valid_paths:
    print(path)

# Step 3: Add new columns for metrics
df_copy['Total_Loss'] = [[] for _ in range(len(df_copy))]
df_copy['Recon_Loss'] = [[] for _ in range(len(df_copy))]
df_copy['BCE_Loss'] = [[] for _ in range(len(df_copy))]
df_copy['KLD'] = [[] for _ in range(len(df_copy))]
df_copy['Accuracy'] = [[] for _ in range(len(df_copy))]
df_copy['Precision'] = [[] for _ in range(len(df_copy))]
df_copy['Recall'] = [[] for _ in range(len(df_copy))]
df_copy['F1_Score'] = [[] for _ in range(len(df_copy))]

# Step 4: Create an instance of text2mcVAEDataset
variables = text2mcPredictor()

variables.embeddings = json.load(open(variables.EMBEDDINGS_FILE))
variables.block2tok = json.load(open(variables.BLOCK_TO_TOK))

# Create an instance of text2mcVAEDataset
dataset = metircsVAEDataset(
    file_paths=valid_paths,
    block2embedding=variables.embeddings,
    block2tok=variables.block2tok,
    block_ignore_list=[102],
    fixed_size=(64, 64, 64)
)

print("Started processing the files")

# Create a DataLoader
batch_size = 1  # Example batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage of the DataLoader
for batch in data_loader:
    data, data_tokens, file_name = batch
    # Extract the file name from the tuple
    file_name = file_name[0]
    
    # Print the file path for debugging
    print(f"Processing file: {file_name}")
    
    # Check if the file path is valid
    if not file_name or not os.path.exists(file_name):
        print(f"Invalid file path: {file_name}")
        continue

    data = data.to(variables.device)
    data_tokens = data_tokens.to(variables.device)

    z, mu, logvar = variables.encoder(data)

    embeddings_pred, block_air_pred = variables.decoder(z)

    total_loss, recon_loss, bce_loss, KLD = loss_function(embeddings_pred, block_air_pred, data, mu, logvar, data_tokens, air_token_id=102)

    block_air_pred_probs = block_air_pred.squeeze(1)  # Shape: (Batch_Size, D, H, W)

    block_air_pred_labels = (block_air_pred_probs >= 0.5).long()
    block_air_labels = (data_tokens != air_token_id).long()

    # Move tensors to the same device
    block_air_pred_labels = block_air_pred_labels.to(variables.device)
    block_air_labels = block_air_labels.to(variables.device)

    # Flatten tensors
    block_air_pred_flat = block_air_pred_labels.view(-1).cpu()
    block_air_labels_flat = block_air_labels.view(-1).cpu()

    # Compute classification metrics
    accuracy = (block_air_pred_flat == block_air_labels_flat).sum().item() / block_air_labels_flat.numel()
    precision = precision_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)
    recall = recall_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)
    f1 = f1_score(block_air_labels_flat, block_air_pred_flat, average='binary', zero_division=0)

    # Update the DataFrame with the metrics total_loss, recon_loss, bce_loss, KLD, accuracy, precision, recall, and f1

    # Normalize file name for comparison
    normalized_file_name = os.path.basename(file_name).lower().strip()

    # Get the index of the current batch
    matching_indices = df_copy[df_copy['PROCESSED_PATHS'].apply(lambda x: any(normalized_file_name.replace('.h5', ext) in os.path.basename(p).lower().strip() for p in x for ext in ['.h5', '.schem', '.schematic']))].index
    if len(matching_indices) > 0:
        idx = matching_indices[0]
        df_copy.at[idx, 'Total_Loss'].append(float(total_loss.item()))
        df_copy.at[idx, 'Recon_Loss'].append(float(recon_loss.item()))
        df_copy.at[idx, 'BCE_Loss'].append(float(bce_loss.item()))
        df_copy.at[idx, 'KLD'].append(float(KLD.item()))
        df_copy.at[idx, 'Accuracy'].append(float(accuracy))
        df_copy.at[idx, 'Precision'].append(float(precision))
        df_copy.at[idx, 'Recall'].append(float(recall))
        df_copy.at[idx, 'F1_Score'].append(float(f1))
    else:
        print(f"No matching index found for file: {file_name}")

# Remove empty lists from the rows for the columns that were created
for column in ["PROCESSED_PATHS",'Total_Loss', 'Recon_Loss', 'BCE_Loss', 'KLD', 'Accuracy', 'Precision', 'Recall', 'F1_Score']:
    df_copy[column] = df_copy[column].apply(lambda x: None if x is None or (isinstance(x, (str, list)) and len(x) == 0) else x)

# Step 5: Save the modified DataFrame to a new CSV file
output_csv_file_path = '/lustre/fs1/groups/jaedo/generative_model/test_metrics.csv'

# Ensure the directory exists
os.makedirs(os.path.dirname(output_csv_file_path), exist_ok=True)

df_copy.to_csv(output_csv_file_path, index=False)