import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist

from utils.common import info_print, save_weights
from utils.pruning_algorithm import patterns


def __acc_cnt(y_hat: torch.Tensor, y: torch.Tensor):
    y_pred = y_hat.argmax(dim=1)
    cmp = y_pred == y
    return float(cmp.sum())


def eval_model(net: nn.Module, data_loader, device, rank=None):
    net.eval()

    if rank is not None:
        num_acc, num_all = torch.zeros(1).to(device), torch.zeros(1).to(device)
    else:
        num_acc, num_all = .0, .0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            num_acc += __acc_cnt(y_hat, y)
            num_all += len(y)

    if rank is not None:
        # waiting for all threads
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        num_acc = reduce_value(num_acc)
        num_all = reduce_value(num_all)

    return num_acc / num_all


def reduce_value(value):
    with torch.no_grad():
        dist.all_reduce(value)
    return value


def pattern_prune(net, device):
    pattern_cnts = torch.zeros(len(patterns), dtype=torch.int, device=device)
    target_patterns = net.action

    i = 0
    if net.name.startswith('vit'):
        for _, module in net.named_modules():
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
        for _, module in net.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            lyr_target_patterns = target_patterns[i].view(-1)  # tensor(dtype=torch.int8, len=本层卷积核数)

            lyr_pattern_cnts = torch.bincount(lyr_target_patterns)
            pattern_cnts += F.pad(lyr_pattern_cnts, (0, len(patterns) - lyr_pattern_cnts.shape[0]), 'constant', 0)

            lyr_patterns_list = []
            for target_pattern in lyr_target_patterns:
                lyr_patterns_list.append(patterns[target_pattern][0].expand(int(module.in_channels / module.groups), -1, -1).to(device))
            lyr_patterns = torch.stack(lyr_patterns_list)  # shape same as module.weight
            if module.weight.data.shape[2] == 3:
                module.weight.data *= lyr_patterns
            i += 1


def finetune_model(path, net: nn.Module, train_loader, val_loader, num_epochs, lr, device):
    print(f"\nStart fine-tuning {net.name} ...")

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
    milestones = [100, 150, 180, 260]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    acc_list = []
    best_val_acc = .0

    for epoch in tqdm(range(num_epochs)):
        net.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()

            # 反向传播后对模型参数应用模式
            pattern_prune(net, device)

            optimizer.step()

        val_acc = eval_model(net, val_loader, device)
        acc_list.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_weights(path, net, best_val_acc)

        scheduler.step()

    info_print(f"accuracy after fine-tuning: {best_val_acc:.4f}")
    print('Finished.')

    return acc_list
