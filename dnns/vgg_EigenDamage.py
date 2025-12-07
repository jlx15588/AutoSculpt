"""
https://github.com/alecwangcq/EigenDamage-Pytorch/blob/master/models/vgg.py
"""

import math
import torch
import torch.nn as nn

from utils.dnn_helper import *

# from utils.common_utils import try_contiguous
# from utils.prune_utils import register_bottleneck_layer, update_QQ_dict
# from utils.prune_utils import LinearLayerRotation, ConvLayerRotation
# from layers.bottleneck_layers import LinearBottleneck, Conv2dBottleneck
# from models.resnet import _weights_init

_AFFINE = True
# _AFFINE = False

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, name='vgg19', dataset='cifar100', depth=19, init_weights=True, cfg=None):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.name = name
        self.action = []
        self.feature = self.make_layers(cfg, True)
        self.dataset = dataset
        if dataset == 'cifar10' or dataset == 'cinic-10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self.apply(weights_init)
        #    self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=_AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
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


def vgg19():
    return VGG()
