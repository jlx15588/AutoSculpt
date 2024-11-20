import os
import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune
import torch.distributed as dist

from utils.pruning_algorithm import patterns


def info_print(content: str):
    blueblod = '\033[1;44m'
    endflag = '\033[0m'
    print(f"\n{blueblod}INFO{endflag} {content}")


def get_per_lyr_ratio(net, action):
    per_layer_ratio = []
    if net.name.startswith('resnet'):
        per_layer_ratio.append(action[0])
        j = 1
        for i in range(1, net.get_info()[0]):
            if i % 2 == 0:
                per_layer_ratio.append(action[0])
            else:
                per_layer_ratio.append(action[j])
                j += 1

    return np.array(per_layer_ratio, dtype=np.float32)


def get_per_lyr_ratio2(net, action):
    """
    :param net:
    :param action: list[tensor], tensor shape: [1, kernels]
    :return:
    """
    per_layer_ratio = []
    per_lyr_action = []

    if net.name == "resnet50":
        per_lyr_action = action
    elif net.name.startswith('resnet'):
        concat_action = action[0]
        per_lyr_action.append(concat_action)
        j = 1
        for i in range(1, net.get_info()[0]):
            if (i - 2) % (net.get_info()[3] * 2) == 0:
                concat_action = action[j]

            if i % 2 == 0:
                per_lyr_action.append(concat_action)
            else:
                per_lyr_action.append(action[j])
                j += 1
    else:
        per_lyr_action = action

    per_lyr_action_new = []
    if not net.name.startswith('vit'):
        j = 0
        for _, module in net.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            important_scores = prune._compute_norm(module.weight.data, 2, 0)  # shape(kernels,)
            important_pos = important_scores.sort().indices
            old_pats = per_lyr_action[j].view(-1).sort().values
            new_pats = torch.ones_like(important_scores, dtype=torch.int)
            for i, pos in enumerate(important_pos):
                new_pats[pos] = old_pats[i]
            per_lyr_action_new.append(new_pats)
            j += 1
    else:
        per_lyr_action_new = per_lyr_action

    for lyr_target_patterns in per_lyr_action_new:
        ratio_sum = .0
        for target_pattern in lyr_target_patterns.view(-1):
            ratio_sum += patterns[target_pattern][1]
        per_layer_ratio.append(ratio_sum / lyr_target_patterns.shape[0])

    return np.array(per_layer_ratio, dtype=np.float32), per_lyr_action_new


def save_weights(path, net: nn.Module, acc):
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, f'{net.name}_{float(acc):.4f}.pt')
    torch.save(net.state_dict(), path)


def get_proper_name(root: str = '.', _name: str = '', suffix: str = None) -> str:

    alike_dirs = []
    for file in os.listdir(root):
        if file.startswith(_name):
            if suffix is not None:
                alike_dirs.append(file.removesuffix(suffix))
            else:
                alike_dirs.append(file)

    if len(alike_dirs) != 0:
        indices = list(map(lambda x: int(x.split('-')[-1]), alike_dirs))
        indices = sorted(indices)
        proper_name = f'{_name}-{indices[-1] + 1}'
    else:
        proper_name = f'{_name}-1'
    return proper_name


def draw_weights(net: nn.Module, compress_ratio, acc):
    import graphviz

    f = graphviz.Digraph('Model Pruning', filename=f'{net.name}_{compress_ratio}_{acc}.gv')
    node_cur = 0
    f.attr('node', shape='circle')

    empty_cnt = 0
    total_cnt = 0

    if net.name.startswith("resnet"):
        for _, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                for i in range(module.out_channels):
                    if module.weight[i].sum() == 0:
                        f.attr('node', shape='circle', style='filled', color='#BBFFFF')
                        f.edge(str(node_cur), str(node_cur + i + 1))
                        f.edge(str(node_cur + i + 1), str(node_cur + module.out_channels + 1))
                        empty_cnt += 1
                    else:
                        f.attr('node', shape='circle', style='', color='black')
                        f.edge(str(node_cur), str(node_cur + i + 1))
                        f.edge(str(node_cur + i + 1), str(node_cur + module.out_channels + 1))
                node_cur += module.out_channels + 1

                total_cnt += module.out_channels

    f.view()


def init_distributed_params(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'distributed init (rank {args.rank}): {args.dist_url}')
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def save_result(path, res: list):
    res_str = ''
    for re in res:
        res_str += f'{re}\n'
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(res_str)


if __name__ == '__main__':
    # name = get_proper_name('..', 'aaa')
    # print(name)

    save_result('./example', [2.45, 45.32, 34.12, 12.55])
