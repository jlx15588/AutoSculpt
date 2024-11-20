import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class LinearLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(LinearLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)

        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1).fill_(self.bias)], 1)
        return x @ self.rotation_matrix

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return "in_features=%s, out_features=%s, trainable=%s" % (self.rotation_matrix.size(1),
                                                                  self.rotation_matrix.size(0),
                                                                  self.trainable)


class ConvLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(ConvLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3)  # out_dim * in_dim
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)
        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        # x: batch_size * in_dim * w * h
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1, x.size(2), x.size(3)).fill_(self.bias)], 1)
        return F.conv2d(x, self.rotation_matrix, None, _pair(1), _pair(0), _pair(1), 1)

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return "in_channels=%s, out_channels=%s, trainable=%s" % (self.rotation_matrix.size(1),
                                                                  self.rotation_matrix.size(0),
                                                                  self.trainable)


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    elif isinstance(m, LinearLayerRotation):
        if m.trainable:
            print('* init Linear rotation')
            init.kaiming_normal(m.rotation_matrix)
    elif isinstance(m, ConvLayerRotation):
        if m.trainable:
            print('* init Conv rotation')
            init.kaiming_normal(m.rotation_matrix)
