import torch.nn as nn
from torchvision import models
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # nn.Conv2d的groups https://blog.csdn.net/qq_44166630/article/details/127802567
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, name, n_class):
        super(MobileNet, self).__init__()
        self.name = name
        self.action = []

        in_planes = 32
        cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

        self.conv1 = conv_bn(3, in_planes, stride=2)

        self.features = self._make_layers(in_planes, cfg, conv_dw)

        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)  # global average pooling

        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_info(self):
        num_conv_layers = 0
        in_channels, out_channels = [], []
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                num_conv_layers += 1
                in_channels.append(module.in_channels)
                out_channels.append(module.out_channels)
        return num_conv_layers, in_channels, out_channels


class MobileNetV2(nn.Module):
    def __init__(self, name, weights):
        super(MobileNetV2, self).__init__()
        self.name = name
        self.action = []
        self.model = models.mobilenet_v2(weights=weights)

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


def mobilenet_v1():
    return MobileNet("mobilenet_v1", 1000)


def mobilenet_v2():
    return MobileNetV2("mobilenet_v2", weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
