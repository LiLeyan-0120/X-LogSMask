
"""Define GNN-Transformer network for edge regression
"""
import torch
import torch.nn as nn

from .layers import TransformerBlock
from .utils import multi_graph, compute_normal_adjacency


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.src_embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, dropout, dim_feedforward) for _ in range(num_layers)])
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)  # Regression task outputs a single value
        )

    def forward(self, X, graph, pe=None):
        src_embedded = self.src_embedding(X.float())
        adjacency = compute_normal_adjacency(graph)
        if pe is not None:
            src_embedded += self.embedding_lap_pos_enc(pe)

        # Encoder
        multigraphs = multi_graph(adjacency, self.num_heads)

        # Encoder
        for layer in self.encoder_layers:
            src_embedded = layer(src_embedded, src_embedded, src_embedded, graph=multigraphs)

        out = self.output(src_embedded)

        return out

    def predict_edge_weights(self, z, edge_index):
        """
        Predict edge weight values

        Args:
            z: node embeddings, shape [num_nodes, feature_dim]
            edge_index: edge indices, shape (2, num_edges), each row contains source and target node indices respectively
        Returns:
            weights: predicted edge weights, shape (num_edges, 1)
        """
        z = z.squeeze(0)  # Remove possible batch dimension, ensure shape is [num_nodes, feature_dim]

        src_nodes = edge_index[0].long()
        dst_nodes = edge_index[1].long()
        u = z[src_nodes]
        v = z[dst_nodes]

        # Concatenate source and target node embeddings
        concat_emb = torch.cat([u, v], dim=-1)

        # Predict weights through regressor
        weights = self.regressor(concat_emb)

        return weights  # (num_edges, edge_attr)
