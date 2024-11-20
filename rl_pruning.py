import os
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

import torch
from torch import nn
import numpy as np

from agents.common import ReplayBuffer, Memory, Memory2
from utils.model_helper import get_model
from utils.dataset_helper import get_dataset
from utils.common import info_print, get_proper_name, init_distributed_params
from env.environment import Environment
from agents.agent import Agent, PPO, PPO2


def parse_args():
    parser = argparse.ArgumentParser(description="RL pruning")

    # datasets
    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10 or cifar100 or imagenet')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--train_ratio', default=0.8, type=float, help='')
    parser.add_argument('--test_ratio', default=1.0, type=float, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--dist-url', default='env://', type=str, help='')

    # dnns
    parser.add_argument('--model', default='resnet110', type=str, help='')

    # training
    parser.add_argument('--device', default="cuda:3", type=str, help="cpu/cuda")
    parser.add_argument('--ckpt_path', default='./checkpoints/pruned_dnns', type=str, help='path to save model weights')
    parser.add_argument('--num_episodes', default=1000, type=int, help='')
    parser.add_argument('--buffer_size', default=500, type=int, help='replaybuffer size')
    parser.add_argument('--minimal_size', default=50, type=int, help='')
    parser.add_argument('--rpbf_batch_size', default=64, type=int, help='')
    # environment
    parser.add_argument('--compress_ratio', default=0.55, type=float, help='')
    parser.add_argument('--pruned_delta_err', default=0.05, type=float, help='')
    parser.add_argument('--num_ft_epochs', default=5, type=int, help='')
    parser.add_argument('--ft_lr', default=0.001, type=float, help='')
    # agent
    parser.add_argument('--num_in_feats', default=64, type=int, help='')
    parser.add_argument('--num_hidden_feats', default=128, type=int, help='')
    parser.add_argument('--num_embed_feats', default=128, type=int, help='')
    parser.add_argument('--num_hiddens', default=256, type=int, help='')
    parser.add_argument('--actor_lr', default=0.001, type=float, help='')
    parser.add_argument('--critic_lr', default=0.001, type=float, help='')
    parser.add_argument('--gamma', default=0.9, type=float, help='折扣因子γ')
    parser.add_argument('--lmbda', default=0.9, type=float, help='')
    parser.add_argument('--agt_upt_epochs', default=3, type=int, help='')
    parser.add_argument('--eps', default=0.2, type=float, help='')

    return parser.parse_args()


def train(env: Environment, agent, num_episodes, minimal_size):
    """pattern pruning training"""
    memory = Memory2()
    return_list = []
    for i in range(num_episodes):
        info_print(f'episode {i + 1}:')
        episode_return = 0
        state = env.reset()
        done = False
        cnt = 0
        while not done:
            action, prob, log_prob = agent.take_action(state)
            next_state, reward, done = env.step2(action)
            if cnt >= 500:
                reward = -100
                done = True
            memory.add(state, np.array(prob.cpu()), log_prob, reward, done)
            state = next_state
            episode_return += reward
            cnt += 1

            if memory.size() > minimal_size:
                print(f'Start update agent ...', end=' ')
                b_s, b_p, b_lp, b_r, b_d = memory.pop()
                transition_dict = {'states': b_s, 'probs': np.array(b_p).squeeze(axis=1), 'log_probs': b_lp, 'rewards': b_r, 'dones': b_d}
                agent.update2(transition_dict)
                memory.clear()
                print('Finished.')

        print(f'totally {cnt} steps.')
        return_list.append(episode_return)

    return return_list


def main():
    args = parse_args()

    rank = None
    if args.distributed:
        init_distributed_params(args)
        rank = args.rank

    device = torch.device(args.device)

    # load pruning DNN
    net = get_model(args.model, device)
    # print(net)

    if args.distributed:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        net.name = net.module.name
        net.get_info = net.module.get_info

    # load dataset
    data_loaders = get_dataset(args.dataset, args.batch_size, args.train_ratio, args.test_ratio, args.distributed)

    # environment
    root = os.path.abspath(args.ckpt_path)
    dir_name = f'{net.name}-{args.compress_ratio}'
    dir_name = get_proper_name(root, dir_name)
    path = os.path.join(root, dir_name)

    env = Environment(
        net, data_loaders, args.compress_ratio, args.pruned_delta_err, args.num_in_feats,
        args.num_ft_epochs, args.ft_lr, path, device, rank
    )

    if net.name.startswith('resnet') and net.name != 'resnet50':
        num_actions = []
        out_chnls = net.get_info()[2]
        num_actions.append(out_chnls[0])
        for i in range(1, net.get_info()[0]):
            if i % 2 != 0:
                num_actions.append(out_chnls[i])
    elif net.name.startswith('vit'):
        num_actions = net.get_info()[1]
    else:
        num_actions = net.get_info()[2]

    # agent
    agent = PPO2(
        args.num_in_feats, args.num_hidden_feats, args.num_embed_feats, args.num_hiddens, num_actions,
        args.actor_lr, args.critic_lr, args.gamma, args.lmbda, args.agt_upt_epochs, args.eps, device
    )

    # training
    returns = train(env, agent, args.num_episodes, args.minimal_size)


if __name__ == "__main__":
    main()
