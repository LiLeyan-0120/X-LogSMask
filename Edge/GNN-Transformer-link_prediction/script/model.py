"""Define GNN-Transformer network
"""
import torch
import torch.nn as nn

from .layers import TransformerBlock
from .utils import multi_graph, compute_normal_adjacency


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.src_embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, dropout, dim_feedforward) for _ in range(num_layers)])
        self.output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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

    def predict_links(self, z, edge_index, predictor_type='lorentz'):
        """
        Link prediction combining the second input form with multiple computation principles

        Args:
            z: Node embeddings, shape [num_nodes, feature_dim]
            edge_index: Edge indices, shape (2, num_edges), each row is source and target node indices respectively
            predictor_type: str, Predictor type, options are 'dot', 'lorentz', 'bilinear', 'mlp'
        Returns:
            scores: Edge scores, shape (num_edges,)
        """
        z = z.squeeze(0)  # Remove possible batch dimension, ensure shape is [num_nodes, feature_dim]

        src_nodes = edge_index[0].long()
        dst_nodes = edge_index[1].long()
        u = z[src_nodes]
        v = z[dst_nodes]

        if predictor_type == "dot":
            # Dot product
            scores = (u * v).sum(dim=-1)
        elif predictor_type == "lorentz":
            n = u.size(1)
            scores = (u[:, :n // 2] * v[:, :n // 2]).sum(dim=-1) - (u[:, n // 2:] * v[:, n // 2:]).sum(dim=-1)
        elif predictor_type == "bilinear":
            # Bilinear mapping u^T W v
            if not hasattr(self, 'bilinear_weight'):
                self.bilinear_weight = nn.Parameter(torch.randn(u.size(1), u.size(1), device=u.device))
                self.register_parameter('bilinear_weight', self.bilinear_weight)
            scores = torch.einsum('bi,ij,bj->b', u, self.bilinear_weight, v)
        elif predictor_type == "mlp":
            # MLP concatenation
            concat_emb = torch.cat([u, v], dim=-1)
            scores = self.predictor(concat_emb).squeeze(-1)
        else:
            raise ValueError(f"Unsupported predictor type: {predictor_type}")

        return scores  # (num_edges,)
