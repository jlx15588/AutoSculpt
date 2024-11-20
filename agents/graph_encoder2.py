import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool

from agents.multi_stage_gcn import MultiStageGCNConv
from utils.pruning_algorithm import patterns


class GraphEncoder(nn.Module):
    def __init__(self, num_in_feats, num_hidden_feats, num_embed_feats, device):
        super(GraphEncoder, self).__init__()
        self.device = device
        self.encoder = Stage2GCNEncoder(num_in_feats, num_hidden_feats, num_embed_feats)

        self.type_features = torch.randn((len(patterns) + 3, num_in_feats), requires_grad=True).to(device)

    def forward(self, graphs):
        if not isinstance(graphs, (tuple, list)):
            graphs = [graphs]

        graph_embeddings = []
        for i, graph in enumerate(graphs):
            graph.edge_features = create_edge_features(graph.edge_type, self.type_features)

            graph_embedding, node_embeddings = self.encoder(graph)

            if i == 0:
                graph_embeddings = graph_embedding
            else:
                graph_embeddings = torch.cat((graph_embeddings, graph_embedding), dim=0)
            # graph_embeddings.append(graph_embedding.cpu().detach().numpy())

        # return torch.tensor(np.array(graph_embeddings)).squeeze(dim=1).to(self.device)
        return graph_embeddings


class Stage2GCNEncoder(nn.Module):
    def __init__(self, num_in_feats, num_hidden_feats, num_out_feats):
        super(Stage2GCNEncoder, self).__init__()
        # self.conv = MultiStageGCNConv(num_in_feats, num_hidden_feats)
        self.conv = GATv2Conv(num_in_feats, num_hidden_feats, heads=8, concat=False, edge_dim=num_in_feats)
        self.linear = nn.Linear(num_hidden_feats, num_out_feats)

    def forward(self, graph):
        x, edge_index, edge_features, batch = graph.x, graph.edge_index, graph.edge_features, graph.batch
        x = F.tanh(self.conv(x, edge_index, edge_features))
        x = F.dropout(x, training=self.training)
        node_embeddings = x

        graph_embeddings = global_mean_pool(x, batch)
        graph_embeddings = F.tanh(self.linear(graph_embeddings))
        return graph_embeddings, node_embeddings


def create_edge_features(edge_types, type_features):
    """

    Args:
        edge_types:
        type_features:
    Returns:
        edge_features:
    """
    edge_features = None
    for i in range(len(edge_types)):
        if i == 0:
            edge_features = type_features[edge_types[i]]
        else:
            edge_features = torch.cat((edge_features, type_features[edge_types[i]]), dim=0)
    return edge_features
