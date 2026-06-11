"""Utility functions
"""


import os
import random
import yaml
import torch
from collections import namedtuple
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt


def create_dir(dir_path):
    # Create folder
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


# Define data structure
Data = namedtuple(
    typename='Data',
    field_names=[
        'X',           # Node features
        'y',           # Node class labels
        'adjacency',   # Adjacency matrix
        'test_mask',   # Test set sample mask
        'train_mask',  # Training set sample mask
        'valid_mask'   # Validation set sample mask
    ]
)


# Define preprocessed data structure
PrepData = namedtuple(
    typename='PrepData',
    field_names=[
        'X',              # Node features
        'y',              # Node class labels
        'edges',          # Edge list
        'adjacency',      # Adjacency matrix
        'test_index',     # Test set sample indices
        'train_index',    # Training set sample indices
        'valid_index'     # Validation set sample indices
        # 'pe'               # Positional encoding
    ]
)


# Load global configuration
def load_config(config_file):
    """Load global configuration

        Load model parameters and training hyperparameters for training models on different datasets

    """

    with open(config_file, 'r', encoding='utf-8') as f:
        # Read yaml file content
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def multi_graph(graph, i):
    # Initialize tensor list and add the original adjacency matrix
    graph_list = [graph]

    # Compute matrix powers sequentially
    for power in range(2, i+1):
        graph_n = torch.matmul(graph_list[-1], graph)
        # graph_n = compute_normal_adjacency(graph_n)   # Normalization is no longer needed, can be derived through degree normalization
        graph_list.append(graph_n)

    # Stack all tensors together to form a tensor with shape (b, h, n, n)
    multigraphs = torch.stack(graph_list, dim=1)

    return multigraphs


def multi_graph_v2(graph, i):
    # Multi-graph structure with fully connected graph version
    full_one_matrix = torch.ones(graph.shape, device=graph.device)
    full_one_matrix = compute_normal_adjacency(full_one_matrix)
    graph_list = [full_one_matrix, graph]

    # Compute matrix powers sequentially
    for power in range(2, i):
        graph_n = torch.matmul(graph_list[-1], graph)
        graph_list.append(graph_n)

    # Stack all tensors together to form a tensor with shape (b, h, n, n)
    multigraphs = torch.stack(graph_list, dim=1)

    return multigraphs


def multi_graph_v3(graph, i):
    # Define multi-graph structure as 1 for connections with multiple paths, 0 for others
    graph_list = [graph]

    # Compute matrix powers sequentially
    for power in range(2, i+1):
        graph_n = torch.matmul(graph_list[-1], graph)
        graph_list.append(graph_n)

    # Stack all tensors together to form a tensor with shape (b, h, n, n)
    multigraphs = torch.stack(graph_list, dim=1)
    binary_multigraphs = (multigraphs != 0).float()

    return binary_multigraphs


def compute_normal_adjacency(batch_adjacency):
    """Batch higher-order adjacency matrix normalization"""

    # Compute degree matrix D (keeping batch dimension)
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


def zero_out_random_elements(graph, num_elements):

    tensor = copy.deepcopy(graph)
    # Get positions of all non-zero elements
    nonzero_indices = torch.nonzero(tensor, as_tuple=False)

    if num_elements > nonzero_indices.size(0):
        raise ValueError("num_elements should be less than or equal to the number of non-zero elements")

    # Randomly select a certain number of positions
    selected_indices = nonzero_indices[torch.randperm(nonzero_indices.size(0))[:num_elements]]

    # Set values at selected positions to 0
    tensor[selected_indices[:, 0], selected_indices[:, 1]] = 0

    return tensor


def set_random_zero_elements_to_value(graph, num_elements, value=0.25):
    tensor = copy.deepcopy(graph)
    # Get positions of all zero elements
    zero_indices = torch.nonzero(tensor == 0, as_tuple=False)

    if num_elements > zero_indices.size(0):
        raise ValueError("num_elements should be less than or equal to the number of zero elements")

    # Randomly select a certain number of positions
    selected_indices = zero_indices[torch.randperm(zero_indices.size(0))[:num_elements]]

    # Set values at selected positions to specified value
    tensor[selected_indices[:, 0], selected_indices[:, 1]] = value

    return tensor


def graph_sampling(x, adjacency, labels, train_index, num_nodes, batch_size):
    """
    Random graph sampling
    Input:
        x(num_nodes, feature): node features
        adjacency: sparse adjacency matrix
        labels: node labels
        train_index: training set labels
        num_nodes: number of subgraph nodes to sample
        batch_size: batch size
    Output:
        batch subgraph node features
        batch subgraph sparse adjacency matrices
        batch subgraph node labels
        training set mask
    """

    batch_X = []
    batch_adjacency = []
    batch_labels = []
    # batch_pe = []
    train_inds = torch.zeros((batch_size, num_nodes))
    train_index = train_index.cpu()
    for i in range(batch_size):
        node_index = random.sample(range(0, x.shape[0]), num_nodes)
        mask = torch.isin(torch.tensor(node_index), train_index)
        train_inds[i, :].masked_fill_(mask, 1)
        batch_X.append(x[node_index, :])
        batch_adjacency.append(adjacency[node_index, :][:, node_index])
        batch_labels.append(labels[node_index])
        # batch_pe.append(pe[node_index])

    return torch.stack(batch_X), torch.stack(batch_adjacency), torch.stack(batch_labels), train_inds.to(x.device)


def graph_sampling_extranode(x, adjacency, labels, train_index, num_nodes, batch_size, special_feature=None):
    """
    Random graph sampling with virtual nodes
    Output:
        batch subgraph features (including virtual nodes)
        batch subgraph adjacency matrices (including virtual node connections)
        batch subgraph labels (virtual node labels set to -1)
        training set mask (virtual node positions set to 0)
    """
    batch_X = []
    batch_adjacency = []
    batch_labels = []
    train_inds = torch.zeros((batch_size, num_nodes + 1))  # +1 for special node

    # Initialize virtual node features
    special_feat = torch.zeros(x.shape[1]) if special_feature is None else special_feature
    special_feat = special_feat.to(x.device)

    for i in range(batch_size):
        # 1. Original sampling
        node_index = random.sample(range(x.shape[0]), num_nodes)
        mask = torch.isin(torch.tensor(node_index), train_index.cpu())

        # 2. Add virtual node
        extended_X = torch.cat([x[node_index], special_feat.unsqueeze(0)], dim=0)

        # 3. Build extended adjacency matrix
        orig_adj = adjacency[node_index, :][:, node_index]
        new_adj = torch.zeros((num_nodes + 1, num_nodes + 1),
                              dtype=orig_adj.dtype, device=orig_adj.device)
        new_adj[:num_nodes, :num_nodes] = orig_adj
        new_adj[num_nodes, :] = 1  # Virtual node connects to all nodes
        new_adj[:, num_nodes] = 1  # Bidirectional connection

        # 4. Handle labels and mask
        extended_labels = torch.cat([
            labels[node_index],
            torch.tensor([7], device=labels.device)
        ])
        train_inds[i, :num_nodes].masked_fill_(mask, 1)  # Virtual node positions remain 0

        batch_X.append(extended_X)
        batch_adjacency.append(new_adj)
        batch_labels.append(extended_labels)

    return torch.stack(batch_X), torch.stack(batch_adjacency), torch.stack(batch_labels), train_inds.to(x.device)


def random_walk_sampling(x, adjacency, labels, train_index, num_nodes, batch_size, walk_length=5):
    device = x.device
    batch_X, batch_adj, batch_labels, train_inds = [], [], [], torch.zeros((batch_size, num_nodes), device=device)

    for i in range(batch_size):
        # Randomly select starting node
        start_node = torch.randint(0, x.size(0), (1,)).item()
        node_index = [start_node]

        # Random walk
        current_node = start_node
        for _ in range(num_nodes - 1):
            neighbors = torch.where(adjacency[current_node] > 0)[0]  # Get neighbor nodes
            if len(neighbors) > 0:
                current_node = neighbors[torch.randint(0, len(neighbors), (1,))].item()
                node_index.append(current_node)
                if len(node_index) >= num_nodes:
                    break

        # Supplement with random nodes if insufficient
        if len(node_index) < num_nodes:
            remaining = num_nodes - len(node_index)
            node_index += torch.randperm(x.size(0))[:remaining].tolist()
        node_index = node_index[:num_nodes]
        node_index = torch.tensor(node_index, device=device)

        # Generate subgraph
        train_inds[i] = torch.isin(node_index, train_index.to(device)).float()
        batch_X.append(x[node_index])
        batch_adj.append(adjacency[node_index][:, node_index])
        batch_labels.append(labels[node_index])

    return torch.stack(batch_X), torch.stack(batch_adj), torch.stack(batch_labels), train_inds


def importance_sampling(x, adjacency, labels, train_index, num_nodes, batch_size, method='degree'):
    device = x.device
    batch_X, batch_adj, batch_labels, train_inds = [], [], [], torch.zeros((batch_size, num_nodes), device=device)

    # Compute node importance (using degree as example)
    if method == 'degree':
        importance = adjacency.sum(dim=1)  # Node degree
    elif method == 'pagerank':
        # Simple PageRank computation (actual implementation can use libraries)
        damping = 0.85
        pr = torch.ones(x.size(0), device=device) / x.size(0)
        for _ in range(10):
            pr = (1 - damping) / x.size(0) + damping * (adjacency @ (pr / adjacency.sum(dim=1).clamp(min=1)))
        importance = pr

    # Sample by importance probability
    prob = importance.float() / importance.sum()
    for i in range(batch_size):
        node_index = torch.multinomial(prob, num_nodes, replacement=False)
        train_inds[i] = torch.isin(node_index, train_index.to(device)).float()
        batch_X.append(x[node_index])
        batch_adj.append(adjacency[node_index][:, node_index])
        batch_labels.append(labels[node_index])

    return torch.stack(batch_X), torch.stack(batch_adj), torch.stack(batch_labels), train_inds


def saints_sampling(x, adjacency, labels, train_index, num_nodes, batch_size, coverage=0.1):
    device = x.device
    batch_X, batch_adj, batch_labels, train_inds = [], [], [], torch.zeros((batch_size, num_nodes), device=device)

    # Pre-compute coverage target (each round must cover at least coverage ratio of nodes)
    target_coverage = int(coverage * x.size(0))
    covered_nodes = torch.zeros(x.size(0), dtype=torch.bool, device=device)

    for i in range(batch_size):
        # Prioritize uncovered nodes
        uncovered = torch.where(~covered_nodes)[0]
        if len(uncovered) > 0 and torch.rand(1) < 0.7:  # 70% probability to select uncovered nodes
            start_node = uncovered[torch.randint(0, len(uncovered), (1,))]
        else:
            start_node = torch.randint(0, x.size(0), (1,))

        # Breadth-first search (BFS) to expand subgraph
        node_index = [start_node.item()]
        queue = [start_node.item()]
        while len(queue) > 0 and len(node_index) < num_nodes:
            current = queue.pop(0)
            neighbors = torch.where(adjacency[current] > 0)[0].tolist()
            neighbors = [n for n in neighbors if n not in node_index]
            node_index.extend(neighbors[:num_nodes - len(node_index)])
            queue.extend(neighbors)

        # Supplement with random nodes if insufficient
        if len(node_index) < num_nodes:
            node_index += torch.randperm(x.size(0))[:num_nodes - len(node_index)].tolist()
        node_index = torch.tensor(node_index[:num_nodes], device=device)

        # Update coverage status
        covered_nodes[node_index] = True
        if covered_nodes.sum() >= target_coverage:
            covered_nodes.fill_(False)  # Reset coverage status

        # Generate subgraph
        train_inds[i] = torch.isin(node_index, train_index.to(device)).float()
        batch_X.append(x[node_index])
        batch_adj.append(adjacency[node_index][:, node_index])
        batch_labels.append(labels[node_index])

    return torch.stack(batch_X), torch.stack(batch_adj), torch.stack(batch_labels), train_inds


def hybrid_sampling(x, adjacency, labels, train_index, num_nodes, batch_size, walk_ratio=0.5):
    if torch.rand(1) < walk_ratio:
        return random_walk_sampling(x, adjacency, labels, train_index, num_nodes, batch_size)
    else:
        return importance_sampling(x, adjacency, labels, train_index, num_nodes, batch_size, method='degree')


def laplacian_positional_encoding(adj, k=10, eps=1e-5):
    # Symmetric normalized Laplacian matrix
    degree = adj.sum(1)
    degree_inv_sqrt = torch.pow(degree + eps, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    norm_adj = torch.diag(degree_inv_sqrt) @ adj @ torch.diag(degree_inv_sqrt).to(adj.device)
    L_sym = torch.eye(adj.shape[0]).to(adj.device)
    L_sym = L_sym - norm_adj

    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(L_sym)  # Automatically sorted
    return eigvecs[:, 1:k+1]  # Return eigenvectors corresponding to the first k smallest eigenvalues


def visualize_attention_heads(pre_graph_energy, post_graph_energy, attention, sample_idx=0, save_path=None):
    """
    Display attention matrix comparison of the first 4 heads in one figure, with support for saving vector graphics

    Parameters:
    - pre_graph_energy: energy matrix before adding graph [batch_size, heads, query_len, key_len]
    - post_graph_energy: energy matrix after adding graph [batch_size, heads, query_len, key_len]
    - attention: final attention matrix [batch_size, heads, query_len, key_len]
    - sample_idx: index of the sample to visualize
    - save_path: vector graphics save path (e.g., "attention_plot.pdf"), None to skip saving
    """
    # Set global font to Times New Roman
    mpl.rcParams["font.family"] = ["Times New Roman", "serif"]
    mpl.rcParams["font.serif"] = ["Times New Roman"]  # Ensure serif font prioritizes Times New Roman
    mpl.rcParams["axes.titlesize"] = 12  # Subplot title font size
    mpl.rcParams["figure.titlesize"] = 16  # Main title font size

    num_heads = min(4, pre_graph_energy.shape[1])  # Display at most 4 heads

    # Create subplots (2 rows, 4 columns)
    fig, axes = plt.subplots(2, num_heads, figsize=(4 * num_heads, 8))

    # Adjust axes shape if only 1 head
    if num_heads == 1:
        axes = axes.reshape(2, 1)

    # Set main title
    fig.suptitle('Energy Matrix with/without MH-LSM', fontsize=16)

    # Iterate over the first 4 heads
    for head_idx in range(num_heads):
        # Get data for specific head and sample
        pre_energy_head = pre_graph_energy[sample_idx, head_idx].cpu().numpy()
        post_energy_head = post_graph_energy[sample_idx, head_idx].cpu().numpy()

        # Draw energy matrix before adding graph (first row)
        im1 = axes[0, head_idx].imshow(pre_energy_head, cmap='viridis')
        axes[0, head_idx].set_title(f'Head {head_idx} - Energy Matrix')
        axes[0, head_idx].set_xticks([])
        axes[0, head_idx].set_yticks([])
        plt.colorbar(im1, ax=axes[0, head_idx], shrink=0.6)

        # Draw energy matrix after adding graph (second row)
        im2 = axes[1, head_idx].imshow(post_energy_head, cmap='viridis')
        axes[1, head_idx].set_title(f'Head {head_idx} - Energy Matrix with MH-LSM')
        axes[1, head_idx].set_xticks([])
        axes[1, head_idx].set_yticks([])
        plt.colorbar(im2, ax=axes[1, head_idx], shrink=0.6)

    # Hide extra subplots if fewer than 4 heads
    for i in range(num_heads, 4):
        if num_heads < 4 and i < 4:  # Avoid index out of bounds
            axes[0, i].set_visible(False)
            axes[1, i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save vector graphics
    if save_path is not None:
        # Recommended vector format
        plt.savefig(save_path, format=None, dpi=300, bbox_inches='tight')
        print(f"Vector graphic saved to: {save_path}")

    plt.show()
