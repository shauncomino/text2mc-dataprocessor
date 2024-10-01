import torch
from torch import nn
from torch.nn import functional as F
from decoder import (
    text2mcVAEAttentionBlock as text2mcVAEAttentionBlock,
    text2mcVAEResidualBlock as text2mcVAEResidualBlock,
)


class text2mcVAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv3d(32, 128, kernel_size=3, padding=1),

            # Downsample
            text2mcVAEResidualBlock(128, 128),
            text2mcVAEResidualBlock(128, 128),
            # Second layer: First downsampling, reduce dimensions by half
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            text2mcVAEResidualBlock(128, 256),
            text2mcVAEResidualBlock(256, 256),
            # Third layer: Second downsampling, reduce dimensions by half again
            nn.Conv3d(
                256, 256, kernel_size=3, stride=2, padding=1
            ),
           
            text2mcVAEResidualBlock(256, 512),
            text2mcVAEResidualBlock(512, 512),
            # Fourth layer: Third downsampling, reduce dimensions by half again
            nn.Conv3d(
                512, 512, kernel_size=3, stride=2, padding=1
            ),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEAttentionBlock(512),
            text2mcVAEResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            # Final layers to adjust the feature map size without changing spatial dimensions
            nn.Conv3d(512, 8, kernel_size=3, padding=1),
            nn.Conv3d(8, 8, kernel_size=1, padding=0),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x):
        # x: (Batch_Size, Channel, Height, Width, Depth)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)
        for module in self:
            x = module(x)

        # Clip the mean and variance from the output
        mu, logvar = torch.chunk(x, 2, dim=1)

        # Ensure the variance lies within reasonable bounds
        logvar = torch.clamp(logvar, -30, 20)

        # Reparameterization trick! (Don't ask me why this works. I do not know.)
        z = self.reparameterize(mu, logvar)

        # Stolen artifact from external GitHub repo (*Indiana Jones theme music*)
        z *= 0.18215
        return z, mu, logvar

