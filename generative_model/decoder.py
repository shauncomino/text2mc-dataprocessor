# decoder.py

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention3D


class text2mcVAEAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention3D(1, channels)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)
        n, c, d, h, w = x.shape
        x = x.view(n, c, d * h * w)
        x = x.transpose(1, 2)
        x = self.attention(x)
        x = x.transpose(1, 2).view(n, c, d, h, w)
        x += residue
        return x


class text2mcVAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)


class text2mcVAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_layers = nn.Sequential(
            nn.Conv3d(8, 1024, kernel_size=3, padding=1),
            text2mcVAEResidualBlock(1024, 1024),
            text2mcVAEAttentionBlock(1024),
            text2mcVAEResidualBlock(1024, 1024),
            text2mcVAEResidualBlock(1024, 1024),
            text2mcVAEResidualBlock(1024, 1024),
        )
        self.upsampling_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(1024, 512, kernel_size=3, padding=1),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            text2mcVAEResidualBlock(256, 256),
            text2mcVAEResidualBlock(256, 256),
            text2mcVAEResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            text2mcVAEResidualBlock(128, 128),
            text2mcVAEResidualBlock(128, 128),
            text2mcVAEResidualBlock(128, 128),
        )
        # Shared layers before splitting
        self.shared_layers = nn.Sequential(
            nn.GroupNorm(32, 128),
            text2mcVAEResidualBlock(128, 128),
        )
        # First head: embeddings
        self.embedding_head = nn.Sequential(
            nn.Conv3d(128, 32, kernel_size=3, padding=1),
            # No activation function, outputs embeddings
        )
        # Second head: block vs. air prediction
        self.block_air_head = nn.Sequential(
            nn.Conv3d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Outputs probability between 0 and 1
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.upsampling_layers(x)
        x = self.shared_layers(x)
        embeddings = self.embedding_head(x)  # Shape: (Batch_Size, 32, D, H, W)
        block_air_prob = self.block_air_head(x)  # Shape: (Batch_Size, 1, D, H, W)
        return embeddings, block_air_prob
