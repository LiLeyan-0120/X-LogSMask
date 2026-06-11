
import torch.nn as nn
from .layers import TransformerBlock, Readout
from .utils import multi_graph, multi_graph_v2, multi_graph_v3, laplacian_positional_encoding, compute_normal_adjacency


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, output_dim, num_layers, dim_feedforward, dropout, aggregate):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.src_embedding = nn.Linear(input_dim, hidden_dim)
        # self.embedding_lap_pos_enc = nn.Linear(15, hidden_dim)
        self.encoder_layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, dropout, dim_feedforward) for _ in range(num_layers)])
        self.fc_out1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = Readout(aggregate)

    def forward(self, x0, adjacency, pe=None):

        src_embedded = self.src_embedding(x0)

        if pe is not None:
            src_embedded += self.embedding_lap_pos_enc(pe)

        multigraphs = multi_graph(adjacency, self.num_heads)

        # Encoder
        for layer in self.encoder_layers:
            src_embedded = layer(src_embedded, src_embedded, src_embedded, graph=multigraphs)

        # Output layer
        readout = self.readout(src_embedded)
        out = self.fc_out1(readout)

        return out
