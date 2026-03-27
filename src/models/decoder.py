import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, latent_dim=512, voxel_size=32):
        super().__init__()

        self.deconv = nn.Sequential(
            # Input: [B, 128, 4, 4, 4]
            nn.ConvTranspose3d(128, 128, 4, 2, 1, bias=False),  # 8³
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),  # 16³
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, 4, 2, 1, bias=False),   # 32³
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # 🔥 ResBlock refinement for thin structures
            ResBlock3D(32, 32),
            ResBlock3D(32, 16),
            ResBlock3D(16, 16),

            nn.Conv3d(16, 1, 1)
        )

    def forward(self, x):
        # Extrude 2D spatial features [B, 512, 4, 4] into 3D grid [B, 128, 4, 4, 4]
        # Depth dimension is formed by folding the 512 channel capacity natively over bounds
        B, C, H, W = x.shape
        x = x.view(B, 128, 4, H, W)
        x = self.deconv(x)
        return x