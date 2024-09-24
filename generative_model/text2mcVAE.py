import torch
from torch import nn
from torch.nn import functional as F
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder

class text2mcVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = text2mcVAEEncoder()
        self.decoder = text2mcVAEDecoder()

    def forward(self, x):
        # The noise for the VAE's reparameterization trick
        noise = torch.randn_like(x)

        # Encoder step: returns latent space representation
        latent = self.encoder(x, noise)

        # Splitting the channels into mean and log variance for VAE
        mu, logvar = torch.chunk(latent, 2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # Reparameterization trick to sample from latent space
        z = mu + eps * std

        # Decoder step: returns reconstructed output
        recon_x = self.decoder(z)
        return recon_x, mu, logvar