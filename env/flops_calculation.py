import copy
import numpy as np
from torch import nn


def __get_per_layer_flops(conv_layer, X):
    X_hidden = conv_layer(X)

    c_in = X.shape[1]
    c_out = X_hidden.shape[1]
    h_out = X_hidden.shape[2]
    w_out = X_hidden.shape[3]
    kernel_h, kernel_w = conv_layer.kernel_size

    flops = h_out * w_out * (c_in * (2 * kernel_h * kernel_w - 1) + 1) * c_out / conv_layer.groups
    return flops, X_hidden


def get_flops(net: nn.Module, train_loader, device):
    X_in, _ = next(iter(train_loader))
    test_case = X_in[0].unsqueeze(dim=0).to(device)  # torch.Size([1, 3, 32, 32]) for cifar10

    flops = []
    if net.name.startswith('vit'):
        for _, module in net.named_modules():
            if isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
                flops.append(1)
    else:
        for _, module in net.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == test_case.shape[1]:
                flop, test_case = __get_per_layer_flops(module, test_case)
                flops.append(flop)

    return flops


def get_pruned_flops(per_layer_flops, pruned_ratio):
    flops = copy.deepcopy(per_layer_flops)

    flops[0] = per_layer_flops[0] * pruned_ratio[0]
    for i in range(1, len(flops)):
        flops[i] = flops[i] * (pruned_ratio[i - 1] + pruned_ratio[i] - pruned_ratio[i - 1] * pruned_ratio[i])

    return flops


def get_pruned_flops2(per_layer_flops, pruned_ratio):
    flops = copy.deepcopy(per_layer_flops)
    flops = np.array(flops)
    return flops * pruned_ratio
