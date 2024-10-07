# encoder.py

import torch
from torch import nn
from torch.nn import functional as F
from decoder import (
    text2mcVAEAttentionBlock as text2mcVAEAttentionBlock,
    text2mcVAEResidualBlock as text2mcVAEResidualBlock,
)

class text2mcVAEEncoder(nn.Module):
    def __init__(self, num_tokens, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim, padding_idx=0)
        self.initial_layers = nn.Sequential(
            nn.Conv3d(embedding_dim, 128, kernel_size=3, padding=1),
            text2mcVAEResidualBlock(128, 128),
            text2mcVAEResidualBlock(128, 128),
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            text2mcVAEResidualBlock(128, 256),
            text2mcVAEResidualBlock(256, 256),
            nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1),
            text2mcVAEResidualBlock(256, 512),
            text2mcVAEResidualBlock(512, 512),
            nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEAttentionBlock(512),
            text2mcVAEResidualBlock(512, 512),
        )
        self.groupnorm = nn.GroupNorm(32, 512)
        # Separate convolutional layers for mu and logvar
        self.mu_conv = nn.Conv3d(512, 4, kernel_size=3, padding=1)
        self.logvar_conv = nn.Conv3d(512, 4, kernel_size=3, padding=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_tokens):
        # x_tokens: (Batch_Size, D, H, W)
        # Embed tokens
        x = self.embedding(x_tokens)  # (Batch_Size, D, H, W, Embedding_Dim)
        # Move embedding dimension to channel dimension
        x = x.permute(0, 4, 1, 2, 3)  # (Batch_Size, Embedding_Dim, D, H, W)
        x = self.initial_layers(x)
        x = self.groupnorm(x)
        mu = self.mu_conv(x)
        logvar = self.logvar_conv(x)
        logvar = torch.clamp(logvar, -20, 20)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
