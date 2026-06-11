import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ViT_XLogSMask.utils.graph_utils import image_to_graph, compute_normal_adjacency, multi_graph, multi_graph_extra

class PatchEmbedding(nn.Module):
    """Image patch embedding layer"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Linear projection layer
        self.projection = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):

        # Convert image to patches
        patches, adjacency = image_to_graph(x, self.patch_size)

        # Linear projection
        x = self.projection(patches)  # [B, N, embed_dim]

        return x, adjacency

class MultiHeadAttentionWithXLogSMask(nn.Module):
    """Multi-head attention mechanism with X-LogSMask"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, values, keys, queries, graph=None):
        B, N, _ = queries.shape

        # Linear projections
        values = self.values(values).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.keys(keys).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.queries(queries).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Incorporate graph information
        if graph is not None:
            # graph shape: [B, num_heads, N, N]
            energy += torch.log(graph.clamp(min=1e-30))

        attention = torch.softmax(energy, dim=-1)
        attention = self.attn_drop(attention)

        out = torch.matmul(attention, values)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        out = self.fc_out(out)
        out = self.proj_drop(out)

        return out

class TransformerBlock(nn.Module):
    """Transformer block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1, use_xlogsmask=True):
        super().__init__()
        self.attn = MultiHeadAttentionWithXLogSMask(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.use_xlogsmask = use_xlogsmask

    def forward(self, x, graph=None):
        # Multi-head attention + residual connection
        if self.use_xlogsmask:
            attention = self.attn(x, x, x, graph)
        else:
            attention = self.attn(x, x, x, None)

        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))

        return out

class Readout(nn.Module):
    """Graph readout operation"""
    def __init__(self, method='mean'):
        super(Readout, self).__init__()
        self.method = method

    def forward(self, X):
        """Graph readout operation forward pass

        Args:
            X: tensor, features of all nodes across all graphs (possibly after pooling)

        Returns:
            readout: tensor, graph features obtained by readout operation
        """
        assert self.method in ['sum', 'mean', 'max'], 'uknown aggregation method'
        if self.method == 'sum':
            readout = torch.sum(X, dim=1)
        elif self.method == 'mean':
            readout = torch.mean(X, dim=1)
        else:
            readout, _ = torch.max(X, dim=1)

        return readout

class VisionTransformerWithXLogSMask(nn.Module):
    """Vision Transformer with X-LogSMask"""
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4,
        dropout=0.1,
        use_xlogsmask=True
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_drop = nn.Dropout(dropout)
        self.num_heads = num_heads

        # Positional encoding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, use_xlogsmask)
            for _ in range(num_layers)
        ])

        # Layer normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # Initialize positional encoding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # Image patch embedding
        x, adjacency = self.patch_embed(x)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)

        # Add positional encoding
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        # Compute normalized adjacency matrix
        adjacency = compute_normal_adjacency(adjacency)

        # Create multi-graph structure
        multigraphs = multi_graph(adjacency, self.num_heads)
        multigraphs_extra = multi_graph_extra(multigraphs)

        # Pass through Transformer blocks
        for layer in self.blocks:
            x = layer(x, graph=multigraphs_extra)

        # Layer normalization
        x = self.norm(x)

        logits = self.head(x[:, -1])

        return logits
