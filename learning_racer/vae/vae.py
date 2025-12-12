import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import vgg16  # Removed VGG16_Weights import

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 5, 10)

class VAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=128):  # Updated z_dim to 128
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),  # Input: image_channels x 80 x 160 -> 32 x 40 x 80
            nn.ReLU(),
            nn.Dropout(0.1),  # New: Dropout for regularization
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
            nn.Dropout(0.1),  # New: Dropout for regularization
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32 x 40 x 80
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),  # -> image_channels x 80 x 160
        )
        # Perceptual loss VGG (use pretrained=True for compatibility)
        self.vgg = vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

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
        return torch.sigmoid(x)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar

    def perceptual_loss(self, recon_x, x):
        feat_recon = self.vgg(recon_x)
        feat_x = self.vgg(x)
        return F.mse_loss(feat_recon, feat_x, reduction='mean')

    def loss_fn(self, image, recon, mean, logvar, beta=4.0, perc_weight=0.1):  # Updated with beta and perc_weight
        BCE = F.binary_cross_entropy(recon, image, reduction='sum')
        KL = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        PERC = self.perceptual_loss(recon, image) * recon.numel()  # Scale to match BCE magnitude
        return BCE + beta * KL + perc_weight * PERC
