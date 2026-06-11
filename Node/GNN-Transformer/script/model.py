"""Define GNN-Transformer network
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .layers import TransformerBlock
from .utils import multi_graph, multi_graph_v2, multi_graph_v3, laplacian_positional_encoding


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, output_dim, num_layers, dim_feedforward, dropout):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.src_embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_lap_pos_enc = nn.Linear(8, hidden_dim)
        self.encoder_layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, dropout, dim_feedforward) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, graph, pe=None):

        src_embedded = self.src_embedding(src)
        if pe is not None:
            src_embedded += self.embedding_lap_pos_enc(pe)
        # Encoder
        multigraphs = multi_graph(graph, self.num_heads)

        for layer in self.encoder_layers:
            src_embedded = layer(src_embedded, src_embedded, src_embedded, graph=multigraphs)

        # Output layer
        out = self.fc_out(src_embedded)
        return out.squeeze(0)
