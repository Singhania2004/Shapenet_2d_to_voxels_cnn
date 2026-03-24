import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim=512, voxel_size=32):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),  # 8x8x8
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),   # 16x16x16
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 1, 4, stride=2, padding=1),     # 32x32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4, 4)
        x = self.deconv(x)
        return x