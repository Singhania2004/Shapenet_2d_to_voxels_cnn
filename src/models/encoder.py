import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    ResNet18 encoder with dual-resolution output.

    Outputs BOTH:
      - skip:  [B, 256, 8, 8] from layer3  — high-res local features (finds thin legs)
      - feats: [B, 512, 4, 4] from layer4  — global semantic features

    8×8 gives 4× more spatial cells than the previous 4×4, catching
    chair legs that are only 1-3 pixels wide in the input image.
    """

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        children = list(resnet.children())

        # Shared stem: conv1, bn1, relu, maxpool, layer1, layer2
        self.stem   = nn.Sequential(*children[:6])   # → [B, 128, 16, 16]

        # layer3 → [B, 256, 8, 8]  (skip connection for decoder)
        self.layer3 = children[6]

        # layer4 → [B, 512, 4, 4]  (main feature embedding)
        self.layer4 = children[7]

    def forward(self, x):
        # x: [B, 3, 128, 128]
        x = self.stem(x)            # [B, 128, 16, 16]
        skip = self.layer3(x)       # [B, 256, 8, 8]  ← high-res
        feats = self.layer4(skip)   # [B, 512, 4, 4]  ← semantic
        return feats, skip