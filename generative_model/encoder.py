import torch
from torch import nn
from torch.nn import functional as F
from decoder import (
    text2mcVAEAttentionBlock as text2mcVAEAttentionBlock,
    text2mcVAEResidualBlock as text2mcVAEResidualBlock,
)


class text2mcVAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # First layer: Initial convolution without changing size, just increasing channels
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
                256, 256, kernel_size=3, stride=2, padding=0
            ),
           
            text2mcVAEResidualBlock(256, 512),
            text2mcVAEResidualBlock(512, 512),
            # Fourth layer: Third downsampling, reduce dimensions by half again
            nn.Conv3d(
                512, 512, kernel_size=3, stride=2, padding=0
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

    def forward(self, x):
        x = self.layers(x)

        # Splitting the channels into mean and log variance for VAE
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # Generate noise with the same shape as mean and stdev
        noise = torch.randn_like(mean)

        # Reparameterization trick for VAE
        x = mean + stdev * noise
        x *= 0.18215  # Scaling factor

        return x, mean, log_variance

