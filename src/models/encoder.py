import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.feature_extractor(x)   # [B, 512, 1, 1]
        x = torch.flatten(x, 1)      # [B, 512]
        x = self.fc(x)                  # [B, latent_dim]
        return x