import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet34(pretrained=True)

        # Break into stages
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )  # 64, 32x32

        self.layer1 = resnet.layer1  # 64, 32x32
        self.layer2 = resnet.layer2  # 128, 16x16
        self.layer3 = resnet.layer3  # 256, 8x8
        self.layer4 = resnet.layer4  # 512, 4x4

    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return f1, f2, f3, f4