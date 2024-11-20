import os
import argparse
import torch

from utils.common import info_print, get_proper_name, draw_weights, save_result
from utils.model_helper import get_model
from utils.dataset_helper import get_dataset
from env.reward_calculation import finetune_model, eval_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default="cuda:1", type=str, help="cpu/cuda")
    parser.add_argument('--model', default='resnet110', type=str, help='')
    parser.add_argument('--compress_ratio', default=0.55, type=float, help='')
    parser.add_argument('--run_idx', default=3, type=int, help='')
    parser.add_argument('--acc', default='0.7836', type=str, help='')

    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10 or cifar100 or imagenet')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--train_ratio', default=1.0, type=float, help='')
    parser.add_argument('--test_ratio', default=1.0, type=float, help='')

    parser.add_argument('--ckpt_path', default='./checkpoints/finetuned_dnns', type=str, help='')
    parser.add_argument('--num_ft_epochs', default=200, type=int, help='')
    parser.add_argument('--ft_lr', default=0.001, type=float, help='')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    net = get_model(args.model, device, fine_tune=True, compress_ratio=args.compress_ratio, run_idx=args.run_idx, acc=args.acc)

    data_loaders = get_dataset(args.dataset, args.batch_size, args.train_ratio, args.test_ratio)
    train_loader, val_loader, test_loader = data_loaders

    accuracy = eval_model(net, test_loader, device)
    info_print(f'accuracy before finetuning: {accuracy}.')

    # path to save model weights
    root = os.path.abspath(args.ckpt_path)
    dir_name = f'{net.name}-{args.compress_ratio}-{args.acc}'
    dir_name = get_proper_name(root, dir_name)
    path = os.path.join(root, dir_name)

    # fine-tuning
    acc_list = finetune_model(path, net, train_loader, test_loader, args.num_ft_epochs, args.ft_lr, device)
    save_result(f'{path}/accs', acc_list)


if __name__ == "__main__":
    main()
