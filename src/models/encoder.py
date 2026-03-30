import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()

        resnet = models.resnet34(pretrained=True)

        # Remove avgpool + fc
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Output: [B, 512, 4, 4]

        self.conv = nn.Conv2d(512, 256, kernel_size=1)

        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.features(x)         # [B, 512, 4, 4]
        x = self.conv(x)             # [B, 256, 4, 4]

        x = x.view(x.size(0), -1)    # flatten
        x = self.fc(x)               # [B, latent_dim]

        return x