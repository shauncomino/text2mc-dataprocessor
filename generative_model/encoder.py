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

    def forward(self, x):
        # x: (Batch_Size, Channel, Height, Width, Depth)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)
        print("Encoder")
        for module in self:
            x = module(x)
            print(x.shape)

        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()
        
        noise = torch.randn_like(variance)

        # Transform N(0, 1) -> N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x

