import os
import argparse
import torch
import matplotlib.pyplot as plt

from utils.common import info_print, get_proper_name, draw_weights
from utils.model_helper import get_model
from utils.dataset_helper import get_dataset
from env.reward_calculation import finetune_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default="cuda:3", type=str, help="cpu/cuda")
    parser.add_argument('--model', default='vgg19', type=str, help='')

    parser.add_argument('--dataset', default='cifar100', type=str, help='cifar10 or cifar100 or imagenet')
    parser.add_argument('--batch_size', default=512, type=int, help='')
    parser.add_argument('--train_ratio', default=1.0, type=float, help='')
    parser.add_argument('--test_ratio', default=1.0, type=float, help='')

    parser.add_argument('--weights_path', default='./data/pretrained_weights', type=str, help='')
    parser.add_argument('--num_epochs', default=200, type=int, help='')
    parser.add_argument('--lr', default=0.01, type=float, help='')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    net = get_model(args.model, device)
    print(net)

    data_loaders = get_dataset(args.dataset, args.batch_size, args.train_ratio, args.test_ratio)
    train_loader, val_loader, test_loader = data_loaders

    root = os.path.abspath(args.weights_path)
    path = os.path.join(root, args.dataset)
    acc_list = finetune_model(path, net, train_loader, test_loader, args.num_epochs, args.lr, device)

    fig, ax = plt.subplots()
    ax.plot(acc_list)
    ax.set(xlabel='Epoch', title='Accuracy')
    fig.savefig(f'{path}/{net.name}_accs.png')


if __name__ == '__main__':
    main()
