# decoder.py

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention3D

class text2mcVAEAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention3D(
            1, channels
        )

    def forward(self, x):
        # x: (Batch_Size, Features, Depth, Height, Width)

        residue = x

        x = self.groupnorm(x)

        n, c, d, h, w = x.shape

        # Flatten Depth, Height, and Width into a single dimension to treat each voxel as a sequence element
        x = x.view(n, c, d * h * w)

        # Transpose to (Batch_Size, Sequence_Length, Features) for attention
        x = x.transpose(1, 2)

        # Perform self-attention WITHOUT mask
        x = self.attention(x)

        # Reshape back to original dimensions
        x = x.transpose(1, 2).view(n, c, d, h, w)

        x += residue  # Add the residual connection

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
            self.residual_layer = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

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
    def __init__(self, num_tokens, embedding_dim):
        super().__init__()
        self.initial_layers = nn.Sequential(
            nn.Conv3d(4, 512, kernel_size=3, padding=1),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEAttentionBlock(512),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
        )
        self.upsampling_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
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
        self.final_layers = nn.Sequential(
            nn.GroupNorm(32, 128),
            nn.Conv3d(128, num_tokens, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.upsampling_layers(x)
        x = self.final_layers(x)
        return x  # Output shape: (Batch_Size, num_tokens, D, H, W)
