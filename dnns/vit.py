from torchvision import models
import torch.nn as nn


class ViT_B_16(nn.Module):
    def __init__(self, name, weights):
        super(ViT_B_16, self).__init__()
        self.name = name
        self.action = []
        self.model = models.vit_b_16(weights=weights)

    def forward(self, x):
        return self.model(x)

    def get_info(self):
        num_encoder_layers = 0
        out_channels = []
        for _, module in self.named_modules():
            if isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
                num_encoder_layers += 1
                out_channels.append(1)
        return num_encoder_layers, out_channels


def vit_b_16():
    return ViT_B_16('vit_b_16', weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
