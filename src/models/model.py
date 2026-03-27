import torch
import torch.nn as nn
from src.models.encoder import Encoder
from src.models.decoder import Decoder


class ReconstructionModel(nn.Module):
    """
    Multi-view voxel reconstruction model.

    Training: receives [B, K, 3, H, W] — encodes each of the K views
              independently, then AVERAGES features before decoding.
              This resolves single-view ambiguity (legs on hidden side).

    Inference: can accept either [B, 3, H, W] or [B, K, 3, H, W].
               test.py passes all N views and averages logits externally.
    """

    def __init__(self, config):
        super().__init__()
        voxel_size = config["data"]["voxel_size"]
        self.encoder = Encoder()
        self.decoder = Decoder(voxel_size)

    def encode_views(self, x):
        """
        x: [B, K, 3, H, W]  or  [B, 3, H, W]
        Returns averaged feats [B, 512, 4, 4] and skip [B, 256, 8, 8].
        """
        if x.dim() == 4:
            # Single view
            return self.encoder(x)   # feats, skip

        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)

        feats_flat, skip_flat = self.encoder(x_flat)

        # Average over views
        feats = feats_flat.view(B, K, *feats_flat.shape[1:]).mean(dim=1)
        skip  = skip_flat.view(B, K, *skip_flat.shape[1:]).mean(dim=1)
        return feats, skip

    def forward(self, x):
        feats, skip = self.encode_views(x)
        return self.decoder(feats, skip)