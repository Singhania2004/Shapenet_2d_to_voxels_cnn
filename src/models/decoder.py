import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Project deepest feature to 3D
        self.project = nn.Conv2d(512, 256, kernel_size=1)

        # 3D decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 1),  # 4 → 8
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(128 + 256, 64, 4, 2, 1),  # 8 → 16
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(64 + 128, 32, 4, 2, 1),  # 16 → 32
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.final = nn.Conv3d(32 + 64, 1, kernel_size=1)

    def expand_to_3d(self, x, depth):
        # x: [B, C, H, W] → [B, C, D, H, W]
        x = x.unsqueeze(2)  # add depth
        x = x.repeat(1, 1, depth, 1, 1)
        return x

    def forward(self, f1, f2, f3, f4):
        # Project deepest feature
        x = self.project(f4)  # [B,256,4,4]

        # Expand to 3D (start from 4x4x4)
        x = self.expand_to_3d(x, 4)

        # === Stage 1 === (4 → 8)
        x = self.up1(x)

        f3_3d = self.expand_to_3d(f3, 8)
        x = torch.cat([x, f3_3d], dim=1)

        # === Stage 2 === (8 → 16)
        x = self.up2(x)

        f2_3d = self.expand_to_3d(f2, 16)
        x = torch.cat([x, f2_3d], dim=1)

        # === Stage 3 === (16 → 32)
        x = self.up3(x)

        f1_3d = self.expand_to_3d(f1, 32)
        x = torch.cat([x, f1_3d], dim=1)

        x = self.final(x)

        return x