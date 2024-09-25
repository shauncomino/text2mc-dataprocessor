import torch
from torch import nn
from torch.nn import functional as F
from decoder import (
    text2mcVAEAttentionBlock,
    text2mcVAEResidualBlock,
)

class text2mcVAEEncoder(nn.Module):
    def __init__(self, num_tokens, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim)

        self.model = nn.Sequential(
            nn.Conv3d(embedding_dim, 128, kernel_size=3, padding=1),
            # Downsample
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
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv3d(512, 8, kernel_size=3, padding=1),
            nn.Conv3d(8, 8, kernel_size=1, padding=0),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: (Batch_Size, Depth, Height, Width)
        x = self.embedding(x)  # (Batch_Size, Depth, Height, Width, Embedding_Dim)
        x = x.permute(0, 4, 1, 2, 3)  # (Batch_Size, Embedding_Dim, Depth, Height, Width)
        x = self.model(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        z = self.reparameterize(mu, logvar)
        z *= 0.18215
        return z, mu, logvar
