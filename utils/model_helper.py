import os
import torch
from torch import nn
from torchvision import models

from dnns import resnet, vgg, mobilenet, vgg_EigenDamage, vit


def get_model(model_name: str, device: torch.device, fine_tune=False, compress_ratio=None, run_idx=None, acc=None) -> nn.Module:
    """
    :param acc:
    :param run_idx:
    :param compress_ratio:
    :param fine_tune:
    :param model_name:
    :param device: cpu/cuda
    :return: DNN with pretrained weights
    """
    net, path = None, None
    root_path = "data/pretrained_weights/"

    if model_name.startswith("resnet"):
        """cifar10"""
        net = resnet.__dict__[model_name]()

        if fine_tune:
            path = f'checkpoints/pruned_dnns/{model_name}-{compress_ratio}-{run_idx}/{model_name}_{acc}.pt'
        else:
            path = root_path + "cifar10/" + model_name + ".pth"
        path = os.path.abspath(path)

        try:
            pretrained_weights = torch.load(path, map_location=device)
            if 'state_dict' in pretrained_weights:
                pretrained_weights = pretrained_weights['state_dict']
                pretrained_weights = {k.replace('module.', ''): v for k, v in pretrained_weights.items()}
            net.load_state_dict(pretrained_weights)
        except FileNotFoundError:
            print(f"can not find{net.name}'s outter pretrained weights!")

    elif model_name.startswith("vgg16"):
        """imagenet"""
        net = vgg.__dict__[model_name](weights=models.VGG16_Weights.IMAGENET1K_V1)

        if fine_tune:
            path = f'checkpoints/pruned_dnns/{model_name}-{compress_ratio}-{run_idx}/{model_name}_{acc}.pt'
            path = os.path.abspath(path)
            pretrained_weights = torch.load(path, map_location=device)
            net.load_state_dict(pretrained_weights)

    elif model_name.startswith("vgg19"):
        """cifar100"""
        # net = vgg.__dict__[model_name]()
        net = vgg_EigenDamage.__dict__[model_name]()

        if fine_tune:
            path = f'checkpoints/pruned_dnns/{model_name}-{compress_ratio}-{run_idx}/{model_name}_{acc}.pt'
        else:
            path = root_path + "cifar100/" + model_name + ".pth"
        path = os.path.abspath(path)

        try:
            pretrained_weights = torch.load(path, map_location=device)
            if 'net' in pretrained_weights:
                pretrained_weights = pretrained_weights['net']
            net.load_state_dict(pretrained_weights)
        except FileNotFoundError:
            print("no pretrained weights found... try to train dnn...")

    elif model_name.startswith("mobilenet_v1"):
        """imagenet"""
        net = mobilenet.__dict__[model_name]()

        if fine_tune:
            path = f'checkpoints/pruned_dnns/{model_name}-{compress_ratio}-{run_idx}/{model_name}_{acc}.pt'
        else:
            path = root_path + "imagenet/" + model_name + ".pth"
        path = os.path.abspath(path)
        pretrained_weights = torch.load(path, map_location=device)

        if 'state_dict' in pretrained_weights:
            pretrained_weights = pretrained_weights['state_dict']
            pretrained_weights = {k.replace('module.', ''): v for k, v in pretrained_weights.items()}

        net.load_state_dict(pretrained_weights)

    elif model_name.startswith("mobilenet_v2"):
        """imagenet"""
        net = mobilenet.__dict__[model_name]()

        if fine_tune:
            path = f'checkpoints/pruned_dnns/{model_name}-{compress_ratio}-{run_idx}/{model_name}_{acc}.pt'
            path = os.path.abspath(path)
            pretrained_weights = torch.load(path, map_location=device)
            net.load_state_dict(pretrained_weights)
        path = 'pytorch inner'

    elif model_name.startswith("vit_b_16"):
        """imagenet"""
        net = vit.__dict__[model_name]()

        if fine_tune:
            path = f'checkpoints/pruned_dnns/{model_name}-{compress_ratio}-{run_idx}/{model_name}_{acc}.pt'
            path = os.path.abspath(path)
            pretrained_weights = torch.load(path, map_location=device)
            net.load_state_dict(pretrained_weights)
        path = 'pytorch inner'

    print(path)
    return net.to(device)
