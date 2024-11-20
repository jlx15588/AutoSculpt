import os.path

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler


def get_dataset(dataset, batch_size, train_ratio, test_ratio, distributed=False):
    np.random.seed(2024)
    data_path = 'data/datasets/'
    sampler = SubsetRandomSampler
    train_loader, val_loader, test_loader = None, None, None

    # if dataset in ['cifar10', 'cifar100']:
    if dataset.startswith('cifar'):
        cifar_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        cifar_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar_normalize,
        ])
        cifar_transform_test = transforms.Compose([
            transforms.ToTensor(),
            cifar_normalize,
        ])

        if dataset == 'cifar10':
            train_set = datasets.CIFAR10(root=data_path, train=True, download=False, transform=cifar_transform_train)
            test_set = datasets.CIFAR10(root=data_path, train=False, download=False, transform=cifar_transform_test)
        else:
            train_set = datasets.CIFAR100(root=data_path, train=True, download=True, transform=cifar_transform_train)
            test_set = datasets.CIFAR100(root=data_path, train=False, download=True, transform=cifar_transform_test)

        train_size = int(len(train_set) * train_ratio)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_sampler = sampler(indices[:train_size])
        val_sampler = sampler(indices[train_size:])

        test_size = int(len(test_set) * test_ratio)
        testset_indices = list(range(len(test_set)))
        np.random.shuffle(testset_indices)
        test_sampler = sampler(testset_indices[:test_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
        val_loader = DataLoader(train_set, batch_size=batch_size, sampler=val_sampler, pin_memory=True) if train_ratio < 1.0 else None
        test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, pin_memory=True)

    elif dataset == 'imagenet':
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        input_size = 224
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])

        train_dir = os.path.join(data_path, 'imagenet/train')
        train_set = datasets.ImageFolder(train_dir, transform=transform_train)

        val_dir = os.path.join(data_path, 'imagenet/val')
        test_set = datasets.ImageFolder(val_dir, transform=transform_test)

        if distributed:
            test_sampler = DistributedSampler(test_set)
        else:
            train_size = int(len(train_set) * train_ratio)
            indices = list(range(len(train_set)))
            np.random.shuffle(indices)
            train_sampler = sampler(indices[:train_size])
            val_sampler = sampler(indices[train_size:])

            test_size = int(len(test_set) * test_ratio)
            testset_indices = list(range(len(test_set)))
            np.random.shuffle(testset_indices)
            test_sampler = sampler(testset_indices[:test_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
        val_loader = DataLoader(train_set, batch_size=batch_size, sampler=val_sampler, pin_memory=True) if train_ratio < 1.0 else None
        test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, pin_memory=True)

    return train_loader, val_loader, test_loader
