import os
import copy
import time
import numpy as np
import torch
from torch import nn

from env.flops_calculation import get_flops, get_pruned_flops, get_pruned_flops2
from env.reward_calculation import eval_model, finetune_model
# from env.graph_construction import get_graph
from env.graph_construction2 import get_graph
from utils.common import info_print, get_per_lyr_ratio, get_per_lyr_ratio2, save_weights
from utils.pruning_algorithm import ln_norm_prune, pattern_prune, patterns


class Environment:
    def __init__(self, net: nn.Module, data_loaders: tuple, compress_ratio, pruned_delta_err, feature_size,
                 num_ft_epochs, ft_lr, path, device, rank=None):
        self.dnn = net
        _, self.val_loader, self.test_loader = data_loaders
        self.feature_size = feature_size
        self.num_ft_epochs = num_ft_epochs
        self.ft_lr = ft_lr
        self.path = path
        self.device = device
        self.rank = rank

        self.per_layer_flops = get_flops(self.dnn, self.test_loader, device)  # list, len = conv_layers
        self.total_flops = sum(self.per_layer_flops)
        self.compress_ratio = compress_ratio
        self.pruned_delta_err = pruned_delta_err

        self.num_prune_layers = net.get_info()[0]  # for example, 109 for resnet110
        if net.name.startswith('vit'):
            action_dim = net.get_info()[1]
        else:
            action_dim = net.get_info()[2]
        self.init_action = self.get_init_action(action_dim)

        start = time.time()
        accuracy = eval_model(self.dnn, self.test_loader, device, self.rank)
        info_print(f"{net.name} initial accuracy: {accuracy}, time-comsuming: {time.time() - start:.4f}s.")
        self.best_accuracy = .0

    def reset(self):
        self.done = False
        self.pruned_dnn = copy.deepcopy(self.dnn)
        self.pruned_per_lyr_ratio = np.zeros(self.num_prune_layers)
        self.retain_per_lyr_ratio = np.ones(self.num_prune_layers)
        state = get_graph(self.dnn, self.init_action, self.feature_size, self.device)
        return state

    def step2(self, action):
        # actions: list[tensor], tensor shape: [1, kernels]
        reward = -1

        self.pruned_per_lyr_ratio, action = get_per_lyr_ratio2(self.dnn, action)
        self.retain_per_lyr_ratio = 1 - self.pruned_per_lyr_ratio

        pruned_per_lyr_flops = get_pruned_flops(self.per_layer_flops, self.pruned_per_lyr_ratio)
        pruned_ratio = sum(pruned_per_lyr_flops) / self.total_flops

        self.pruned_dnn = pattern_prune(self.dnn, action, self.device)
        accuracy = eval_model(self.pruned_dnn, self.test_loader, self.device)

        if abs(pruned_ratio - self.compress_ratio) <= self.pruned_delta_err:
            self.done = True


            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                if self.rank is not None and self.rank == 0:
                    save_weights(self.path, self.pruned_dnn, self.best_accuracy)
                else:
                    save_weights(self.path, self.pruned_dnn, self.best_accuracy)

            reward = accuracy * 100

            print(f'accuracy after pruning: {accuracy}, pruning ratio: {pruned_ratio:.4f}', end=', ')
        else:
            reward += accuracy

        state = get_graph(self.pruned_dnn, action, self.feature_size, self.device)
        return state, reward, self.done

    def get_init_action(self, out_channels):
        action = []
        for out_channel in out_channels:
            action.append(
                torch.ones((1, out_channel), dtype=torch.int) * (len(patterns) - 1)
            )
        return action
