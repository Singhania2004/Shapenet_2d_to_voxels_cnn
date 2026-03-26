import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim=512, voxel_size=32):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, 2, 1),  # 8³
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.ConvTranspose3d(256, 128, 4, 2, 1),  # 16³
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, 4, 2, 1),   # 32³
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # 🔥 NEW refinement block
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv3d(32, 1, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4, 4)
        x = self.deconv(x)
        return x