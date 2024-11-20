import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from utils.pruning_algorithm import patterns


def get_graph(net, ratio_or_pats, feature_size, device):

    return _get_graph(net, ratio_or_pats, feature_size).to(device)

    # level2_type_dict = {  # the conv in each layer have dif types, starts from 0
    #     "concatenates": len(out_channels),
    #     "shortCut1": len(out_channels) + 1,
    #     "shortCut2": len(out_channels) + 2,
    #     "bacthNorm": len(out_channels) + 3,
    #     "linear": len(out_channels) + 4,
    #     "ReLu": len(out_channels) + 5
    # }
    #
    # hierarchical_graph = {
    #     'level1': level1_graph(net, in_channels, feature_size).to(device),
    #     'level2': level2_graph(net, level2_type_dict, out_channels, feature_size).to(device)
    # }
    # return hierarchical_graph


def conv_sub_graph(node_cur, n_filter, edge_list, edge_type, conv_type=None, concat_type=None):
    """
    Construct a subgraph for conv operation in a DNN
    Assume the number of filters is 5 in a conv layer:
        /-  1  -\
      /---  2  ---\
    0 ----  3  ---- 6
      \---  4  ---/
        \-  5  -/
    :param node_cur:
    :param n_filter:
    :param edge_list:
    :param edge_type:
    :param conv_type:
    :param concat_type:
    :return:
    """
    for i in range(n_filter):
        edge_list.append([node_cur, node_cur + (i + 1)])
        edge_type.append(conv_type[i])

        edge_list.append([node_cur + (i + 1), node_cur + n_filter + 1])
        edge_type.append(concat_type)

    node_cur += n_filter + 1

    return node_cur


def encoder_sub_graph(node_cur, edge_list: list, edge_type: list, encoder_type, concat_type):
    for i in range(len(encoder_type)):
        edge_list.extend([[node_cur, node_cur + 1], [node_cur, node_cur + 2], [node_cur, node_cur + 3]])
        edge_type.extend([encoder_type[i].view(-1), encoder_type[i].view(-1), encoder_type[i].view(-1)])

        edge_list.extend([[node_cur + 1, node_cur + 4], [node_cur + 2, node_cur + 4], [node_cur + 3, node_cur + 5],
                         [node_cur + 4, node_cur + 5], [node_cur + 5, node_cur + 6], [node_cur + 6, node_cur + 7]])
        edge_type.extend([concat_type] * 6)
        # res-connect
        edge_list.append([node_cur, node_cur + 7])
        edge_type.append(concat_type + 1)
        node_cur += 7
    return node_cur


def _get_graph(net, target_pats, feature_size):
    """

    :param net:
    :param target_pats: list[tensor], tensor shape: [1, kernels]
    :param feature_size:
    :return:
    """
    out_channels = net.get_info()[1] if net.name.startswith('vit') else net.get_info()[2]
    concat_id = len(patterns)

    node_cur = 0
    edge_list = []
    edge_type = []

    if net.name.startswith('vit'):
        # encoders
        node_cur = encoder_sub_graph(node_cur, edge_list, edge_type, target_pats, concat_id)
        # mlp
        edge_list.append([node_cur, node_cur + 1])
        edge_type.append(concat_id + 2)
        node_cur += 1
    else:
        for i, out_channel in enumerate(out_channels):
            node_cur = conv_sub_graph(node_cur, out_channel, edge_list, edge_type, target_pats[i].view(-1), concat_id)

    Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(), edge_type=edge_type)
    num_nodes = node_cur + 1
    Graph.x = torch.randn([num_nodes, feature_size])
    Graph.edge_features = None
    Graph = DataLoader([Graph], batch_size=1, shuffle=False)
    Graph = get_next_graph_batch(Graph)

    return Graph


def conv_sub_graph2(node_cur, n_filter, edge_list, edge_type, conv_type=None, concat_type=None):

    for i in range(n_filter):
        edge_list.append([node_cur, node_cur + (i + 1)])
        edge_type.append(conv_type)

        edge_list.append([node_cur + (i + 1), node_cur + n_filter + 1])
        edge_type.append(concat_type)

    node_cur += n_filter + 1

    return node_cur


def _get_graph2(net, ratio_or_pats, feature_size):

    out_channels = net.get_info()[2]
    concat_id = len(patterns)

    if not isinstance(ratio_or_pats, list):
        out_channels = (out_channels * ratio_or_pats).astype(int)

    node_cur = 0
    edge_list = []
    edge_type = []

    if net.name.startswith("resnet"):
        for i, out_channel in enumerate(out_channels):
            node_cur = conv_sub_graph2(node_cur, out_channel, edge_list, edge_type, 1, concat_id)

    Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(), edge_type=edge_type)
    num_nodes = node_cur + 1
    Graph.x = torch.randn([num_nodes, feature_size])
    Graph.edge_features = None
    Graph = DataLoader([Graph], batch_size=1, shuffle=False)
    Graph = get_next_graph_batch(Graph)

    return Graph


def get_next_graph_batch(data_loader):
    return next(iter(data_loader))
