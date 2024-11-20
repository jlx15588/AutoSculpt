import copy

import torch
from torch import nn
from torch.nn.utils import prune
from torch.nn.utils.prune import BasePruningMethod


'''-----------------------------------------
    an example of predefined 6-patterns set.
'''
patterns = [
    (torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), 1.0),
    (torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0, 1]]), 2.0 / 3),
    (torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1]]), 2.0 / 3),
    (torch.tensor([[0, 1, 1], [0, 1, 1], [0, 1, 1]]), 1.0 / 3),
    (torch.tensor([[0, 0, 0], [1, 1, 1], [1, 1, 1]]), 1.0 / 3),
    (torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), .0)
]


def ln_norm_prune(net, pruned_per_lyr_ratio, n=2):
    """
    :param net:
    :param pruned_per_lyr_ratio:
    :param n: n-order norm
    :return:
    """
    pruned_net = copy.deepcopy(net)

    i = 0
    for _, module in pruned_net.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=pruned_per_lyr_ratio[i], n=n, dim=0)
            prune.remove(module, name='weight')
            i += 1

    return pruned_net


def pattern_prune(net, target_patterns, device):
    """

    :param device:
    :param net:
    :param target_patterns: list[tensor], tensor shape: [1, kernels]
    :return:
    """
    pruned_net = copy.deepcopy(net)

    i = 0
    if net.name.startswith('vit'):
        for _, module in pruned_net.named_modules():
            if not isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
                continue
            target_pat = int(target_patterns[i])
            pattern = torch.ones_like(module.weight.data)
            split_pos = pattern.shape[0] // 3
            if target_pat == 0:
                pattern[:] = 0
            elif target_pat == 1:
                pattern[:, :(split_pos * 2)] = 0
            elif target_pat == 2:
                pattern[:(split_pos * 2), :] = 0
            elif target_pat == 3:
                pattern[:, :split_pos] = 0
            elif target_pat == 4:
                pattern[:split_pos, :] = 0

            module.weight.data *= pattern
            i += 1
    else:
        for _, module in pruned_net.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            lyr_target_patterns = target_patterns[i].view(-1)
            lyr_patterns_list = []
            for target_pattern in lyr_target_patterns:
                lyr_patterns_list.append(patterns[target_pattern][0].expand(int(module.in_channels / module.groups), -1, -1).to(device))
            lyr_patterns = torch.stack(lyr_patterns_list)  # shape same as module.weight
            if module.weight.data.shape[2] == 3:
                module.weight.data *= lyr_patterns
            i += 1

    return pruned_net


if __name__ == '__main__':
    w1 = torch.randn(16, 3, 3, 3)
    w1_norm = prune._compute_norm(w1, 2, 0)
    print()

    # from utils.model_helper import get_model
    #
    # device_ = torch.device('cuda:0')
    # model = get_model('resnet110', device_)
