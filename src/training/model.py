"""
Convolutional Autoencoder for anomaly detection.

Architecture:
  Encoder: compresses input image into a latent representation
  Decoder: reconstructs the image from the latent representation

At inference time, reconstruction error (MSE) is the anomaly score.
High error = likely anomaly.
"""

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # 1x32x32 → 16x16x16
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 16x16x16 → 32x8x8
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 32x8x8 → 64x4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.net = nn.Sequential(
            # 64x4x4 → 32x8x8
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 32x8x8 → 16x16x16
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 16x16x16 → 1x32x32
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 64, 4, 4)
        return self.net(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE — used as anomaly score at inference."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=(1, 2, 3))
