
import torch
import torch.nn as nn

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

    def forward(self, values, keys, queries, graph=None):
        N = queries.shape[0]

        # Linear projections
        values = self.values(values).view(N, -1, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, -1, self.heads, self.head_dim)
        queries = self.queries(queries).view(N, -1, self.heads, self.head_dim)

        # Scaled dot-product attention
        energy = torch.matmul(queries.transpose(1, 2), keys.transpose(1, 2).transpose(2, 3))

        # Incorporate graph information
        if graph is not None:

            # energy += graph.masked_fill(graph == 0, float("-1e10"))  # v1

            # graph = zero_out_random_elements(graph, 100)  # da(cora:100 ) can improve by one point

            energy += torch.log(graph.clamp(min=1e-30))

            # energy += self.spatialEmbedding(graph)

            # energy += 2*torch.log(graph+float("1e-30"))  # v2

            # energy = torch.mul(energy, graph.unsqueeze(0).unsqueeze(0))  # v3: can the method of replacing with minimum value be given by a formula

            # energy = energy.masked_fill(graph.unsqueeze(0).unsqueeze(0) == 0, float("-1e20"))  # v4

            # energy += torch.log(graph.clamp(min=1e-30)) * self.alpha  # v5 try: adding weight has average effect

            # energy = self.parameter(torch.stack((energy, graph), dim=4)).squeeze(-1)  # v6 try: adding weight has average effect

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

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
