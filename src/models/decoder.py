import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, voxel_size=32):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)

        self.deconv = nn.Sequential(
            # 4 → 8
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            # 8 → 16
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # 16 → 32
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        # 🔥 Residual refinement block
        self.refine = nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 1, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4, 4)

        x = self.deconv(x)
        x = self.refine(x)

        return x