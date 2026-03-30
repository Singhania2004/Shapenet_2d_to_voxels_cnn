import torch.nn as nn
from src.models.encoder import Encoder
from src.models.decoder import Decoder


class ReconstructionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)
        voxel = self.decoder(f1, f2, f3, f4)
        return voxel