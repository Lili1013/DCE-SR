import torch
import torch.nn as nn
from loguru import logger

class V_Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, v2e, embed_dim, cuda="cpu"):
        super(V_Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.v2e = v2e
        self.embed_dim = embed_dim
    def forward(self, nodes, to_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            e_v = self.v2e.weight[list(tmp_adj)] # fast: item embedding
            e_v_mean = torch.mean(e_v,dim=0)

            embed_matrix[i] = e_v_mean
        to_feats = embed_matrix

        return to_feats
