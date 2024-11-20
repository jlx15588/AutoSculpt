import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import torch.distributed as dist

from utils.common import info_print, save_weights


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
