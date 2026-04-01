import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)

        self.up1 = nn.ConvTranspose3d(256, 128, 4, 2, 1)
        self.up2 = nn.ConvTranspose3d(128, 64, 4, 2, 1)
        self.up3 = nn.ConvTranspose3d(64, 32, 4, 2, 1)

        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(32)

        self.relu = nn.ReLU()

        # 🔥 Feature projection layers
        self.proj3 = nn.Conv2d(256, 128, 1)
        self.proj2 = nn.Conv2d(128, 64, 1)
        self.proj1 = nn.Conv2d(64, 32, 1)

        self.final = nn.Conv3d(32, 1, 1)

    def inject(self, x, feat2d):
        # Global pooling → scalar guidance
        feat = torch.mean(feat2d, dim=[2,3], keepdim=True)
        feat = feat.unsqueeze(2)  # [B,C,1,1,1]
        return x + feat

    def forward(self, latent, features):
        f1, f2, f3, f4 = features

        x = self.fc(latent)
        x = x.view(-1, 256, 4, 4, 4)

        # 4 → 8
        x = self.relu(self.bn1(self.up1(x)))
        x = self.inject(x, self.proj3(f3))

        # 8 → 16
        x = self.relu(self.bn2(self.up2(x)))
        x = self.inject(x, self.proj2(f2))

        # 16 → 32
        x = self.relu(self.bn3(self.up3(x)))
        x = self.inject(x, self.proj1(f1))

        return self.final(x)