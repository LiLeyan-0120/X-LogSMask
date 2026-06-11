"""Utility functions
"""


import os
import yaml
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_dense_adj

def create_dir(dir_path):
    # Create folder
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return

def load_config(config_file):
    """Load global configuration

        Load model parameters and training hyperparameters for training models on different datasets

    """

    with open(config_file, 'r', encoding='utf-8') as f:
        # Read yaml file content
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def compute_normal_adjacency(batch_adjacency):
    """Batch higher-order adjacency matrix normalization"""

    # Compute degree matrix D (keep batch dimension)
    degree = torch.sum(batch_adjacency, dim=2)  # shape: [batch, n]

    # Compute D^-0.5
    d_hat = torch.pow(degree, -0.5)
    d_hat[torch.isinf(d_hat)] = 0.0

    # Expand to diagonal matrix [batch, n, n]
    eye = torch.eye(batch_adjacency.size(1),
                    device=batch_adjacency.device)
    d_hat = d_hat.unsqueeze(2) * eye.unsqueeze(0)

    # Compute normalized adjacency matrix
    norm_adj = torch.bmm(torch.bmm(d_hat, batch_adjacency), d_hat)

    return norm_adj

def multi_graph(graph, i):
    # Initialize tensor list and add the original adjacency matrix
    graph_list = [graph]

    # Compute matrix powers iteratively
    for power in range(2, i+1):
        graph_n = torch.matmul(graph_list[-1], graph)
        graph_list.append(graph_n)

    # Stack all tensors together, forming a tensor with shape (b, h, n, n)
    multigraphs = torch.stack(graph_list, dim=1)

    return multigraphs

def multi_graph_v2(graph, i):
    # Multi-graph structure with fully connected graph version
    full_one_matrix = torch.ones(graph.shape, device=graph.device)
    full_one_matrix = compute_normal_adjacency(full_one_matrix)
    graph_list = [full_one_matrix, graph]

    # Compute matrix powers iteratively
    for power in range(2, i):
        graph_n = torch.matmul(graph_list[-1], graph)
        graph_list.append(graph_n)

    # Stack all tensors together, forming a tensor with shape (b, h, n, n)
    multigraphs = torch.stack(graph_list, dim=1)

    return multigraphs

def multi_graph_v3(graph, i):
    # Define multi-graph structure where connections with multiple paths are 1, others are 0
    graph_list = [graph]

    # Compute matrix powers iteratively
    for power in range(2, i+1):
        graph_n = torch.matmul(graph_list[-1], graph)
        graph_list.append(graph_n)

    # Stack all tensors together, forming a tensor with shape (b, h, n, n)
    multigraphs = torch.stack(graph_list, dim=1)
    binary_multigraphs = (multigraphs != 0).float()

    return binary_multigraphs


def adj_to_edge_index(adj_matrix: torch.Tensor) -> list[torch.Tensor]:
    """
    Generate edge index list from batch adjacency matrices in reverse

    Input: batch adjacency matrix, shape (batch_size, node_num, node_num)
    Output: edge index list, length = batch_size, each element is a tensor with shape (2, edge_num)
         (edge_num is the number of non-zero edges in the corresponding adjacency matrix, varies per sample)

    Key notes:
    1. The position (i,j) of non-zero elements in the adjacency matrix corresponds to column [i,j] in the edge index
    2. Handles directed graphs by default (for undirected graphs, additional filtering of duplicate edges is needed, see notes below)
    3. Output edge index format conforms to PyTorch Geometric standard (edge_index: 2xE)
    """
    # 1. Validate input shape (must be a 3D tensor of batch x node x node)
    if adj_matrix.dim() != 3:
        raise ValueError(f"Input adjacency matrix must be a 3D tensor (batch, node, node), current shape: {adj_matrix.shape}")

    batch_size = adj_matrix.shape[0]
    edge_index_list = []  # Store edge indices for each batch

    # 2. Iterate over each batch's adjacency matrix to generate corresponding edge indices
    for idx in range(batch_size):
        # Get current batch's adjacency matrix (node x node)
        single_adj = adj_matrix[idx]

        # 3. Find positions of non-zero elements in adjacency matrix -> shape (edge_num, 2), each row is (source node i, target node j)
        # Note: torch.nonzero() returns coordinates of non-zero elements, ordered by (row, col)
        nonzero_pos = single_adj.nonzero(as_tuple=False)  # (edge_num, 2)
        # Filter duplicate edges for undirected graph: only keep edges where i < j
        mask = nonzero_pos[:, 0] < nonzero_pos[:, 1]
        nonzero_pos = nonzero_pos[mask]
        single_edge_index = nonzero_pos.t()  # No duplicate edges at this point

        # 5. Keep device consistency (same device as input adjacency matrix)
        single_edge_index = single_edge_index.to(adj_matrix.device)

        # 6. Add to result list
        edge_index_list.append(single_edge_index)

    return edge_index_list


def hit_ratio_at_k(pos_scores, neg_scores, k):
    """Compute HR@K, accepts flattened tensor as input"""

    if len(pos_scores) == 0:
        return 0.0

    if len(neg_scores) < k:
        return 1.0

    # Get top K highest scores from negative samples, find the lowest among them (Kth place score)
    # Note: topk returns sorted from high to low, [0] is values, [-1] is the Kth place score
    kth_neg_score = torch.topk(neg_scores, k).values[-1]

    # Count positive samples with scores strictly greater than the Kth negative sample score
    hits = torch.sum(pos_scores > kth_neg_score).item()

    return hits / len(pos_scores)


def mean_reciprocal_rank(pos_scores, neg_scores):
    """Compute MRR, accepts flattened tensor as input"""

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0

    # Expand dimensions for broadcast comparison (P,1) vs (1,N) -> (P,N)
    pos = pos_scores.unsqueeze(1)  # Shape (P, 1)
    neg = neg_scores.unsqueeze(0)  # Shape (1, N)

    # Optimistic rank: number of negative samples strictly greater than positive sample (positive sample ranks higher)
    optimistic_rank = (neg > pos).sum(dim=1)  # Shape (P,), count corresponding to each positive sample

    # Pessimistic rank: number of negative samples greater than or equal to positive sample (positive sample ranks lower)
    pessimistic_rank = (neg >= pos).sum(dim=1)  # Shape (P,)

    # Final rank = (optimistic + pessimistic) / 2 + 1 (ranking starts from 1)
    ranking = 0.5 * (optimistic_rank + pessimistic_rank) + 1.0

    # Compute reciprocal rank for each positive sample, take mean
    mrr = (1.0 / ranking).mean().item()

    return mrr
