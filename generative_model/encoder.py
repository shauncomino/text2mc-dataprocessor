# encoder.py

import torch
from torch import nn
from torch.nn import functional as F

class AxialAttention3D(nn.Module):
    def __init__(self, in_channels, heads=8, dim_heads=32):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dim_heads = dim_heads
        self.total_heads_dim = heads * dim_heads

        self.query_conv = nn.Conv1d(in_channels, self.total_heads_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, self.total_heads_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, self.total_heads_dim, kernel_size=1)
        self.out_conv = nn.Conv1d(self.total_heads_dim, in_channels, kernel_size=1)
        self.scale = dim_heads ** -0.5

    def forward(self, x):
        # x shape: (batch_size, in_channels, length)
        batch_size, channels, length = x.size()

        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, self.heads, self.dim_heads, length)
        k = k.view(batch_size, self.heads, self.dim_heads, length)
        v = v.view(batch_size, self.heads, self.dim_heads, length)

        # Compute attention
        q = q.permute(0, 1, 3, 2)  # (batch_size, heads, length, dim_heads)
        k = k.permute(0, 1, 2, 3)  # (batch_size, heads, dim_heads, length)
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Compute output
        v = v.permute(0, 1, 3, 2)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, self.total_heads_dim, length)
        out = self.out_conv(out)
        return out

class AxialAttentionBlock3D(nn.Module):
    def __init__(self, in_channels, heads=8, dim_heads=32):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dim_heads = dim_heads
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.attention_d = AxialAttention3D(in_channels, heads, dim_heads)
        self.attention_h = AxialAttention3D(in_channels, heads, dim_heads)
        self.attention_w = AxialAttention3D(in_channels, heads, dim_heads)

    def forward(self, x):
        # x shape: (batch_size, in_channels, D, H, W)
        batch_size, channels, D, H, W = x.shape
        residue = x

        x = self.groupnorm(x)

        # Attention along depth axis
        x_d = x.permute(0, 3, 4, 2, 1).reshape(-1, channels, D)
        out_d = self.attention_d(x_d)
        out_d = out_d.view(batch_size, H, W, channels, D).permute(0, 3, 4, 1, 2)

        # Attention along height axis
        x_h = x.permute(0, 2, 4, 3, 1).reshape(-1, channels, H)
        out_h = self.attention_h(x_h)
        out_h = out_h.view(batch_size, D, W, channels, H).permute(0, 3, 1, 4, 2)

        # Attention along width axis
        x_w = x.permute(0, 2, 3, 4, 1).reshape(-1, channels, W)
        out_w = self.attention_w(x_w)
        out_w = out_w.view(batch_size, D, H, channels, W).permute(0, 3, 1, 2, 4)

        # Sum the outputs
        out = out_d + out_h + out_w
        out += residue
        return out

class PositionalEncoding3D(nn.Module):
    def __init__(self, D, H, W, channels):
        super().__init__()
        self.D = D
        self.H = H
        self.W = W
        self.channels = channels

        # Create coordinate grids
        z = torch.arange(D).unsqueeze(1).unsqueeze(2).expand(D, H, W)
        y = torch.arange(H).unsqueeze(0).unsqueeze(2).expand(D, H, W)
        x = torch.arange(W).unsqueeze(0).unsqueeze(1).expand(D, H, W)

        # Normalize coordinates
        z = z.float() / D
        y = y.float() / H
        x = x.float() / W

        # Stack coordinates
        self.register_buffer('coords', torch.stack((z, y, x), dim=3).unsqueeze(0))

        self.linear = nn.Linear(3, channels)

    def forward(self, x):
        batch_size = x.size(0)
        coords = self.coords.repeat(batch_size, 1, 1, 1, 1)
        coords = coords.to(x.device)
        coords = coords.view(-1, 3)
        pos_embed = self.linear(coords)
        pos_embed = pos_embed.view(batch_size, self.D, self.H, self.W, self.channels)
        pos_embed = pos_embed.permute(0, 4, 1, 2, 3)
        return pos_embed

class text2mcVAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
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

class text2mcVAEEncoder(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.positional_encoding = None

        self.initial_layers = nn.Sequential(
            nn.Conv3d(embedding_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            text2mcVAEResidualBlock(128, 256),
            text2mcVAEResidualBlock(256, 256),
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
            text2mcVAEResidualBlock(512, 512),
            text2mcVAEResidualBlock(512, 512),
            AxialAttentionBlock3D(512),
            text2mcVAEResidualBlock(512, 1024),
            nn.Conv3d(1024, 1024, kernel_size=3, stride=2, padding=1),
            text2mcVAEResidualBlock(1024, 1024),
        )
        self.groupnorm = nn.GroupNorm(32, 1024)
        self.mu_conv = nn.Conv3d(1024, 8, kernel_size=3, padding=1)
        self.logvar_conv = nn.Conv3d(1024, 8, kernel_size=3, padding=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size, embedding_dim, D, H, W = x.shape
        if self.positional_encoding is None or self.positional_encoding.D != D or self.positional_encoding.H != H or self.positional_encoding.W != W:
            self.positional_encoding = PositionalEncoding3D(D, H, W, embedding_dim).to(x.device)
        x = x + self.positional_encoding(x)
        x = self.initial_layers(x)
        x = self.groupnorm(x)
        mu = self.mu_conv(x)
        logvar = self.logvar_conv(x)
        logvar = torch.clamp(logvar, -20, 20)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
