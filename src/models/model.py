import torch.nn as nn
from src.models.encoder import Encoder
from src.models.decoder import Decoder


class ReconstructionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config["model"]["latent_dim"])
        self.decoder = Decoder(config["model"]["latent_dim"])

    def forward(self, x):
        latent, features = self.encoder(x)
        voxel = self.decoder(latent, features)
        return voxel