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

def loss_function(recon_x, x, mu, logvar, mask):
    # If recon_x has more channels, slice to match x's channels
    recon_x = recon_x[:, :x.size(1), :, :, :]

    # Adjust the mask to match the shape of recon_x
    mask_expanded = mask.unsqueeze(1)  # Add a singleton dimension for channels
    mask_expanded = mask_expanded.expand_as(recon_x)  # Expand to match recon_x's channel size

    # Apply the mask
    recon_x_masked = recon_x * mask_expanded
    x_masked = x * mask_expanded

    # Calculate the Binary Cross-Entropy loss with logits
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x_masked, x_masked, reduction='sum')

    # Calculate KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Initialize the model components, optimizer, and gradient scaler

# Change the following line to the commented line proceeding it when using a capable machine to train
device_type = "cpu"
# device_type = "cuda" if torch.cuda.is_available() else "cpu"


device = torch.device(device_type)
encoder = text2mcVAEEncoder().to(device)
decoder = text2mcVAEDecoder().to(device)


rand_input = torch.zeros(1, 32, 256, 256, 256).to(device_type)
encoded_input = encoder(rand_input)
decoded_output = decoder(encoded_input)



