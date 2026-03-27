import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """3D residual block — preserves thin feature gradients via skip."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class Decoder(nn.Module):
    """
    Spatial 2D→3D decoder with explicit Z-expansion and 8×8 skip connection.

    Architecture:
      4×4 feats → unsqueeze → ConvTranspose3d(depth-only, 1→4) → [B,256,4,4,4]
      8×8 skip  → project to 3D → [B, 64, 8, 8, 8]
         ↓
      [B,256,4,4,4] → upsample 4→8 → concat skip → [B,128,8,8,8] → ResBlock
                    → upsample 8→16 → [B,64,16,16,16] → ResBlock
                    → upsample 16→32 → [B,32,32,32,32] → ResBlock
                    → Conv3d(1×1) → [B,1,32,32,32]
    """

    def __init__(self, voxel_size=32):
        super().__init__()

        # ── Explicit Z-expansion: depth-only ConvTranspose (XY preserved) ──
        # Input [B, 512, 1, 4, 4] → [B, 256, 4, 4, 4]
        self.expand_z = nn.Sequential(
            nn.ConvTranspose3d(512, 256,
                               kernel_size=(4, 1, 1),
                               stride=(4, 1, 1),
                               padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # ── Project 2D 8×8 skip into 3D ──
        # [B, 256, 8, 8] → [B, 64, 8, 8] → unsqueeze+expand → [B, 64, 8, 8, 8]
        self.skip_proj = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Upsample: 4³ → 8³ (then merge 8×8 skip) ──
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        # After concat with skip → 128 + 64 = 192 ch → reduce to 128
        self.merge1 = nn.Sequential(
            nn.Conv3d(192, 128, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock3D(128)

        # ── 8³ → 16³ ──
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.res2 = ResBlock3D(64)

        # ── 16³ → 32³ ──
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.res3 = ResBlock3D(32)

        self.output = nn.Conv3d(32, 1, 1)

    def forward(self, feats, skip_2d):
        # feats:    [B, 512, 4, 4]
        # skip_2d:  [B, 256, 8, 8]

        # ── Explicit Z-expansion ──
        x = feats.unsqueeze(2)               # [B, 512, 1, 4, 4]
        x = self.expand_z(x)                 # [B, 256, 4, 4, 4]

        # ── Prepare 8×8 skip as 3D volume ──
        s = self.skip_proj(skip_2d)          # [B, 64, 8, 8]
        B = s.shape[0]
        s = s.unsqueeze(2).expand(-1, -1, 8, -1, -1)  # [B, 64, 8, 8, 8]

        # ── Upsample 4³→8³, inject skip ──
        x = self.up1(x)                      # [B, 128, 8, 8, 8]
        x = self.merge1(torch.cat([x, s], dim=1))  # [B, 128, 8, 8, 8]
        x = self.res1(x)

        # ── 8³→16³ ──
        x = self.res2(self.up2(x))           # [B, 64, 16, 16, 16]

        # ── 16³→32³ ──
        x = self.res3(self.up3(x))           # [B, 32, 32, 32, 32]

        return self.output(x)                # [B, 1, 32, 32, 32]