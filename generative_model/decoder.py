# decoder.py

import torch
from torch import nn
from torch.nn import functional as F

# Reuse AxialAttention3D and related classes from encoder.py
from encoder import AxialAttentionBlock3D, text2mcVAEResidualBlock

class text2mcVAEDecoder(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.initial_layers = nn.Sequential(
            nn.ConvTranspose3d(8, 1024, kernel_size=4, stride=2, padding=1),
            text2mcVAEResidualBlock(1024, 1024),
            AxialAttentionBlock3D(1024),
            text2mcVAEResidualBlock(1024, 1024),
        )
        self.upsampling_layers = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            text2mcVAEResidualBlock(256, 256),
            text2mcVAEResidualBlock(256, 256),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            text2mcVAEResidualBlock(128, 128),
            text2mcVAEResidualBlock(128, 128),
        )
        self.shared_layers = nn.Sequential(
            nn.GroupNorm(32, 128),
            text2mcVAEResidualBlock(128, 128),
        )
        self.embedding_head = nn.Sequential(
            nn.Conv3d(128, embedding_dim, kernel_size=3, padding=1),
        )
        self.block_air_head = nn.Sequential(
            nn.Conv3d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.upsampling_layers(x)
        x = self.shared_layers(x)
        embeddings = self.embedding_head(x)
        block_air_prob = self.block_air_head(x)
        return embeddings, block_air_prob
