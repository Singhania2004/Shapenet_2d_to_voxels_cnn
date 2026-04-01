import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()

        resnet = models.resnet34(pretrained=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, latent_dim)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.features[0:4](x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        pooled = self.pool(f4).view(x.size(0), -1)
        latent = self.fc(pooled)

        return latent, (f1, f2, f3, f4)