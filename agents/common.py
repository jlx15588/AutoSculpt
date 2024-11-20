import copy
import collections
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agents.graph_encoder2 import GraphEncoder
from utils.pruning_algorithm import patterns


class PolicyNet(nn.Module):
    def __init__(self, num_in_feats, num_hidden_feats, num_embed_feats, num_hiddens, num_actions, device):
        super(PolicyNet, self).__init__()
        self.graph_encoder = GraphEncoder(num_in_feats, num_hidden_feats, num_embed_feats, device)
        self.linear1 = nn.Linear(num_embed_feats, num_hiddens)
        # self.linear_mu = nn.Linear(num_hiddens, num_actions)
        # self.linear_std = nn.Linear(num_hiddens, num_actions)

        # self.pat_chooser = []
        # for num_action in num_actions:
        #     self.pat_chooser.append(nn.Linear(num_hiddens, num_action).to(device))

        self.linear2 = nn.Linear(num_hiddens, len(patterns))

        self.to(device)

    def forward(self, state):
        graph_embedding = self.graph_encoder(state)
        x = F.relu(self.linear1(graph_embedding))

        # mu = F.tanh(self.linear_mu(x))
        # std = F.softplus(self.linear_std(x))
        # return mu, std

        # outs, log_probs = [], []
        # for linear in self.pat_chooser:
        #     out = linear(x)
        #     out_min = out.min(dim=1, keepdim=True).values
        #     out_max = out.max(dim=1, keepdim=True).values
        #     out_normal = (out - out_min) / (out_max - out_min)
        #     log_probs.append(torch.mean(out_normal, dim=1, keepdim=True))
        #     out_normal = torch.round(out_normal * 5).to(torch.int)
        #     outs.append(out_normal.cpu())  # out_normal's shape same as out
        # batch_log_probs = torch.tensor(tuple(zip(*log_probs)), requires_grad=True)
        # return outs, torch.log(torch.mean(batch_log_probs, dim=1))

        # return F.softmax(self.linear2(x), dim=1)
        return F.tanh(self.linear2(x))


class ValueNet(nn.Module):
    def __init__(self, num_in_feats, num_hidden_feats, num_embed_feats, device):
        super(ValueNet, self).__init__()
        self.graph_encoder = GraphEncoder(num_in_feats, num_hidden_feats, num_embed_feats, device)
        self.linear = nn.Linear(num_embed_feats, 1)
        self.to(device)

    def forward(self, state):
        graph_embedding = self.graph_encoder(state)
        value = F.tanh(self.linear(graph_embedding))
        return value


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)


class Memory:
    def __init__(self):
        self.buffer = []

    def add(self, state, action, log_prob, reward, done):
        self.buffer.append((state, action, log_prob, reward, done))

    def size(self):
        return len(self.buffer)

    def pop(self):
        states, actions, log_probs, rewards, dones = zip(*self.buffer)
        self.buffer.clear()
        return states, actions, log_probs, rewards, dones


class Memory2:
    def __init__(self):
        self.buffer = []

    def add(self, *args):
        self.buffer.append(args)

    def size(self):
        return len(self.buffer)

    def pop(self):
        return tuple(zip(*self.buffer))

    def clear(self):
        self.buffer.clear()


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


if __name__ == '__main__':
    replay_buffer = ReplayBuffer(10)
    replay_buffer.add(1, 2.0, 3.5, 4, {'a': 6.5})
    replay_buffer.add(2, 2.6, 3.1, 4, {'a': 6.2})
    replay_buffer.add(3, 2.3, 3.7, 4, {'a': 6.1})
    replay_buffer.add(4, 2.1, 3.2, 4, {'a': 6.7})
    replay_buffer.add(5, 2.5, 3.9, 4, {'a': 6.6})
    s, a, r, ns, d = replay_buffer.sample(2)  # s: tuple
    print()
