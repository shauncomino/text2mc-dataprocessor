import torch
import torch.nn as nn
import torch.optim as optim

class text2mcVAE(nn.Module):
    def __init__(self, embedding_dim=32, latent_dim=128):
        super(text2mcVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(embedding_dim, 32, kernel_size=3, stride=2, padding=1),  # Reduces each dimension by half
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # Again reduces by half
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # Again reduces by half
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4)),  # Reduces everything to a fixed size (4, 4, 4)
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4 * 4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1, 1, 1)),
            nn.ConvTranspose3d(latent_dim, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, embedding_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        print(f"Encoded shape: {encoded.shape}")  # Check the output shape
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar
