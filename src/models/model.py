import torch.nn as nn
from src.models.encoder import Encoder
from src.models.decoder import Decoder


class ReconstructionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        latent_dim = config["model"]["latent_dim"]
        voxel_size = config["data"]["voxel_size"]

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, voxel_size)

    def forward(self, x):
        latent = self.encoder(x)
        voxel = self.decoder(latent)
        return voxel