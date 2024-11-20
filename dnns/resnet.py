import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        """
        in_planes: 16, planes: 32, stride: 2
        :param in_planes:
        :param planes:
        :param stride:
        :param option:
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(

                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        x_bypass = self.shortcut(x)
        out += x_bypass
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, name, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.name = name
        self.num_blocks = num_blocks
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_info(self):
        num_conv_layers = 0
        in_channels, out_channels = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                num_conv_layers += 1
                in_channels.append(module.in_channels)
                out_channels.append(module.out_channels)
        return num_conv_layers, in_channels, out_channels, self.num_blocks[0]


class TorchResNet(nn.Module):
    def __init__(self, name, weights):
        super(TorchResNet, self).__init__()
        self.name = name
        self.model = models.resnet50(weights=weights)

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


def resnet20():
    return ResNet("resnet20", BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet("resnet32", BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet("resnet44", BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet("resnet56", BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet("resnet110", BasicBlock, [18, 18, 18])


def resnet50():
    return TorchResNet("resnet50", weights=models.ResNet50_Weights.IMAGENET1K_V1)
