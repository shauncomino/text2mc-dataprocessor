# decoder.py

import torch
from torch import nn
from torch.nn import functional as F

# Reuse AxialAttentionBlock3D and text2mcVAEResidualBlock from encoder.py
from encoder import AxialAttentionBlock3D, text2mcVAEResidualBlock

class text2mcVAEDecoder(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.initial_layers = nn.Sequential(
            nn.ConvTranspose3d(8, 1024, kernel_size=4, stride=4, padding=0),  # D=8 -> D=32
            text2mcVAEResidualBlock(1024, 1024),
            AxialAttentionBlock3D(1024),
            text2mcVAEResidualBlock(1024, 1024),
        )
        self.upsampling_layers = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2, padding=0),  # D=32 -> D=64
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
        )
        self.shared_layers = nn.Sequential(
            nn.GroupNorm(32, 512),
            text2mcVAEResidualBlock(512, 512),
        )
        # First head: embeddings
        self.embedding_head = nn.Sequential(
            nn.Conv3d(512, embedding_dim, kernel_size=3, padding=1),
        )
        # Second head: block vs. air prediction
        self.block_air_head = nn.Sequential(
            nn.Conv3d(512, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.upsampling_layers(x)
        x = self.shared_layers(x)
        embeddings = self.embedding_head(x)
        block_air_prob = self.block_air_head(x)
        return embeddings, block_air_prob
