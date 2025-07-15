import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    """
    Backbone network for feature extraction.
    Supports ResNet-50 and ResNet-101.
    Outputs feature maps for TLCFS.
    """
    def __init__(self, name='resnet101', pretrained=True, out_dim=256):
        super().__init__()
        if name == 'resnet101':
            net = models.resnet101(pretrained=pretrained)
        elif name == 'resnet50':
            net = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        # Remove fully connected and avgpool layers
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        # Project to out_dim if needed
        self.proj = nn.Conv2d(2048, out_dim, 1)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        return x  # (B, out_dim, H', W') 