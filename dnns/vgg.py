import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, name, weights):
        super(VGG16, self).__init__()
        self.name = name
        self.action = []
        self.model = models.vgg16(weights=weights)

    def forward(self, x):
        return self.model(x)

    def get_info(self):
        num_conv_layers = 0
        in_channels, out_channels = [], []
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                num_conv_layers += 1
                in_channels.append(module.in_channels)
                out_channels.append(module.out_channels)
        return num_conv_layers, in_channels, out_channels


class VGG19(nn.Module):
    def __init__(self, name):
        super(VGG19, self).__init__()
        self.name = name
        self.action = []
        self.model = models.vgg19()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1000, out_features=100)
        )

    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)

    def get_info(self):
        num_conv_layers = 0
        in_channels, out_channels = [], []
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                num_conv_layers += 1
                in_channels.append(module.in_channels)
                out_channels.append(module.out_channels)
        return num_conv_layers, in_channels, out_channels


def vgg16(weights):
    return VGG16("vgg16", weights)


def vgg19():
    return VGG19("vgg19")
