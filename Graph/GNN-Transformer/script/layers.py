"""MinCutPool network layer functions
"""


import torch
import torch.nn as nn
from .utils import multi_graph, zero_out_random_elements, set_random_zero_elements_to_value, visualize_attention_heads


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.keys = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.queries = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = torch.nn.Linear(embed_size, embed_size)
        # self.parameter = nn.Parameter(torch.tensor(0.3))  # Define a 1x1 parameter
        # self.parameter = torch.nn.Linear(2, 1, bias=False)

    def forward(self, values, keys, queries, graph=None):
        N = queries.shape[0]

        # Linear projections
        values = self.values(values).view(N, -1, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, -1, self.heads, self.head_dim)
        queries = self.queries(queries).view(N, -1, self.heads, self.head_dim)
        # values = self.values(values).view(N, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        # keys = self.keys(keys).view(N, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        # queries = self.queries(queries).view(N, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        energy = torch.matmul(queries.transpose(1, 2), keys.transpose(1, 2).transpose(2, 3))

        # Incorporate graph information
        if graph is not None:

            # energy += graph.masked_fill(graph == 0, float("-1e10"))  # v1

            # graph = zero_out_random_elements(graph, 100)  # da(cora:100 ) can improve by one point
            # graph = set_random_zero_elements_to_value(graph, 10000)(poor performance and slow speed)

            # self.n = (graph.sum() / energy.sum()).item()
            # print('graph:energy=', n.item())

            # pre_graph_energy = energy.clone().detach()
            energy += torch.log(graph.clamp(min=1e-30))
            # xlogsmask = torch.log(graph.clamp(min=1e-30)).clone().detach()
            # post_graph_energy = energy.clone().detach()

            # energy += self.spatialEmbedding(graph)

            # energy += 2*torch.log(graph+float("1e-30"))  # v2

            # energy = torch.mul(energy, graph.unsqueeze(0).unsqueeze(0))  # v3: Can the method of replacing with minimum values be given by formula

            # energy = energy.masked_fill(graph.unsqueeze(0).unsqueeze(0) == 0, float("-1e20"))  # v4

            # energy += torch.log(graph.clamp(min=1e-30)) * self.alpha  # v5 try: adding weights has average effect

            # energy = self.parameter(torch.stack((energy, graph), dim=4)).squeeze(-1)  # v6 try: adding weights has average effect

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        # visualize_attention_heads(pre_graph_energy, xlogsmask, post_graph_energy, attention, sample_idx=0, save_path="result/visualize/attention_comparison.svg")

        out = torch.matmul(attention, values.transpose(1, 2)).transpose(1, 2)
        out = self.fc_out(out.contiguous().view(N, -1, self.embed_size))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, graph):
        attention = self.attention(value, key, query, graph)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Readout(nn.Module):
    """Graph readout operation
    """
    def __init__(self, method):
        super(Readout, self).__init__()
        self.method = method

    def forward(self, X):
        """Graph readout operation forward pass
            Inputs:
            -------
            X: tensor, features of all nodes across all graphs (possibly after pooling)

            Output:
            -------
            readout: tensor, graph features obtained from readout operation
            """
        assert self.method in ['sum', 'mean', 'max'], 'uknown aggregation method'
        if self.method == 'sum':
            readout = torch.sum(X, dim=1)
        elif self.method == 'mean':
            readout = torch.mean(X, dim=1)
        else:
            readout, _ = torch.max(X, dim=1)

        return readout
