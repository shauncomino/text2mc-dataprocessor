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
from train import loss_function
from sklearn.metrics import precision_score, recall_score, f1_score


air_token_id = 102

# Step 1: Read the CSV file and make a copy of it
csv_file_path = 'path/to/your/csv_file.csv'
df = pd.read_csv(csv_file_path)
df_copy = df.copy()

# Extract the paths from the 'PROCESSED_PATHS' column
paths_list = df_copy['PROCESSED_PATHS'].apply(lambda x: x.split(','))

# Flatten the list of lists into a single list
all_paths = [path.strip() for sublist in paths_list for path in sublist]

# Step 3: Add new columns for metrics
df_copy['Total_Loss'] = [[] for _ in range(len(df_copy))]
df_copy['Recon_Loss'] = [[] for _ in range(len(df_copy))]
df_copy['BCE_Loss'] = [[] for _ in range(len(df_copy))]
df_copy['KLD'] = [[] for _ in range(len(df_copy))]
df_copy['Accuracy'] = [[] for _ in range(len(df_copy))]
df_copy['Percision'] = [[] for _ in range(len(df_copy))]
df_copy['Recall'] = [[] for _ in range(len(df_copy))]
df_copy['F1_Score'] = [[] for _ in range(len(df_copy))]

# Step 4: Create an instance of text2mcVAEDataset
variables = text2mcPredictor()

variables.embeddings = json.load(open(variables.EMBEDDINGS_FILE))
variables.block2tok = json.load(open(variables.BLOCK_TO_TOK))

# Create an instance of text2mcVAEDataset
dataset = metircsVAEDataset(
    file_paths=all_paths,
    block2embedding=variables.embeddings,
    block2tok=variables.block2tok,
    block_ignore_list=[102],
    fixed_size=(64, 64, 64)
)

# Create a DataLoader
batch_size = 1  # Example batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage of the DataLoader
for data, data_tokens, file_name in data_loader:
    data = data.to(variables.device)

    z, mu, logvar = variables.encoder(data)

    embeddings_pred, block_air_pred = variables.decoder(z)

    total_loss, recon_loss, bce_loss, KLD = loss_function(embeddings_pred, block_air_pred, data, mu, logvar, data_tokens, air_token_id=102)

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

    # Update the DataFrame with the metrics total_loss, recon_loss, bce_loss, KLD, accuracy, precision, recall, and f1

    # Get the index of the current batch
    idx = df_copy[df_copy['PROCESSED_PATHS'].apply(lambda x: file_name[0] in x)].index[0]
    df_copy.at[idx, 'Total_Loss'].append(total_loss.item())
    df_copy.at[idx, 'Recon_Loss'].append(recon_loss.item())
    df_copy.at[idx, 'BCE_Loss'].append(bce_loss.item())
    df_copy.at[idx, 'KLD'].append(KLD.item())
    df_copy.at[idx, 'Accuracy'].append(accuracy)
    df_copy.at[idx, 'Precision'].append(precision)
    df_copy.at[idx, 'Recall'].append(recall)
    df_copy.at[idx, 'F1_Score'].append(f1)

# Step 5: Save the modified DataFrame to a new CSV file
output_csv_file_path = 'path/to/your/output_csv_file.csv'
df_copy.to_csv(output_csv_file_path, index=False)