import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms

class VAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),  # Input: image_channels x 80 x 160 -> 32 x 40 x 80
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> 64 x 20 x 40
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # -> 128 x 10 x 20
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # -> 256 x 5 x 10
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 5 * 10, z_dim)
        self.fc_logvar = nn.Linear(256 * 5 * 10, z_dim)
        self.decoder_input = nn.Linear(z_dim, 256 * 5 * 10)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 256 x 5 x 10 -> 128 x 10 x 20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 64 x 20 x 40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32 x 40 x 80
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),  # -> image_channels x 80 x 160
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)  # Sample z here
        return z, mu, logvar  # Now returns 3 values

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 5, 10)
        x = self.decoder(x)
        return x

    def forward(self, x):
        z, mu, logvar = self.encode(x)  # Updated to unpack 3 values
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_fn(self, image, recon, mean, logvar, beta=1.0):
        REC = F.binary_cross_entropy(recon, image, reduction='sum')
        KL = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return REC + beta * KL
