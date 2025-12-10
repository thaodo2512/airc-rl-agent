import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 5, 10)

class VAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=32):
        super(VAE, self).__init__()
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
            nn.ConvTranspose2d(32, image_channels*2, 4, stride=2, padding=1),  # -> (image_channels*2) x 80 x 160 for mean + logvar
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 5, 10)
        x = self.decoder(x)
        mu_y = x[:, :3, :, :]
        log_sigma_y = x[:, 3:, :, :]
        sigma_y = torch.exp(log_sigma_y)
        return mu_y, sigma_y

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        mu_y, sigma_y = self.decode(z)
        return mu_y, sigma_y, mu, logvar

    def loss_fn(self, image, mu_y, sigma_y, mean, logvar):
        m_vae_loss = (image - mu_y) ** 2 / sigma_y
        m_vae_loss = 0.5 * torch.sum(m_vae_loss)
        a_vae_loss = torch.log(2 * 3.14 * sigma_y)
        a_vae_loss = 0.5 * torch.sum(a_vae_loss)
        KL = -0.5 * torch.sum((1 + logvar - mean.pow(2) - logvar.exp()), dim=0)
        KL = torch.mean(KL)
        return KL + m_vae_loss + a_vae_loss
