import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights  # Added for modern weights

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 5, 10)

class VAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=128):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),  # Input: image_channels x 80 x 160 -> 32 x 40 x 80
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout for regularization
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
            nn.Dropout(0.1),  # Dropout for regularization
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32 x 40 x 80
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),  # -> image_channels x 80 x 160
            nn.Tanh(),  # Changed from Sigmoid to match training (outputs [-1,1])
        )
        # Perceptual loss VGG (updated with weights for modern PyTorch)
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        # ImageNet normalization for VGG
        self.vgg_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])

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
        return x  # No sigmoid; Tanh is in sequential

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar

    def perceptual_loss(self, recon_x, x):
        # Denormalize from [-1,1] to [0,1]
        x = (x + 1) / 2
        recon_x = (recon_x + 1) / 2
        # Apply ImageNet normalization
        x_norm = self.vgg_normalization(x)
        recon_x_norm = self.vgg_normalization(recon_x)
        feat_recon = self.vgg(recon_x_norm)
        feat_x = self.vgg(x_norm)
        return F.mse_loss(feat_recon, feat_x, reduction='mean')

    def loss_fn(self, image, recon, mean, logvar, beta=1.0, perc_weight=0.1):  # Updated beta default to 1.0
        # Changed to MSE for reconstruction (matches updated training)
        REC = F.mse_loss(recon, image, reduction='sum')
        KL = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        PERC = self.perceptual_loss(recon, image) * 10000  # Updated scaling (fixed factor instead of numel())
        return REC + beta * KL + perc_weight * PERC
