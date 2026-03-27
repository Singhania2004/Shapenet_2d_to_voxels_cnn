import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        # Remove AdaptiveAvgPool2d and fully connected layer to preserve spatial dimensions
        # children()[-2] is layer4, outputting [B, 512, 4, 4]
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        x = self.feature_extractor(x)   # [B, 512, 4, 4] (for 128x128 input)
        return x