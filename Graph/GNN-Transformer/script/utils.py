
import os
import yaml
import torch
import copy
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

def create_dir(dir_path):
    # Create directory
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

    # Compute matrix powers sequentially
    for power in range(2, i+1):
        graph_n = torch.matmul(graph_list[-1], graph)
        # graph_n = compute_normal_adjacency(graph_n)   # No need for normalization, can be derived from degree normalization
        graph_list.append(graph_n)

    # Stack all tensors together to form a tensor of shape (b, h, n, n)
    multigraphs = torch.stack(graph_list, dim=1)

    return multigraphs


def multi_graph_v2(graph, i):
    # Add fully-connected graph version of multi-graph structure
    full_one_matrix = torch.ones(graph.shape, device=graph.device)
    full_one_matrix = compute_normal_adjacency(full_one_matrix)
    graph_list = [full_one_matrix, graph]

    # Compute matrix powers sequentially
    for power in range(2, i):
        graph_n = torch.matmul(graph_list[-1], graph)
        graph_list.append(graph_n)

    # Stack all tensors together to form a tensor of shape (b, h, n, n)
    multigraphs = torch.stack(graph_list, dim=1)

    return multigraphs


def multi_graph_v3(graph, i):
    # Define multi-graph structure where connections with multiple paths are 1, others are 0
    graph_list = [graph]

    # Compute matrix powers sequentially
    for power in range(2, i+1):
        graph_n = torch.matmul(graph_list[-1], graph)
        graph_list.append(graph_n)

    # Stack all tensors together to form a tensor of shape (b, h, n, n)
    multigraphs = torch.stack(graph_list, dim=1)
    binary_multigraphs = (multigraphs != 0).float()

    return binary_multigraphs


def zero_out_random_elements(graph, num_elements):

    tensor = copy.deepcopy(graph)
    # Get positions of all non-zero elements
    nonzero_indices = torch.nonzero(tensor, as_tuple=False)

    if num_elements > nonzero_indices.size(0):
        raise ValueError("num_elements should be less than or equal to the number of non-zero elements")

    # Randomly select a certain number of positions
    selected_indices = nonzero_indices[torch.randperm(nonzero_indices.size(0))[:num_elements]]

    # Set the values at selected positions to 0
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

    # Set the values at selected positions to the specified value
    tensor[selected_indices[:, 0], selected_indices[:, 1]] = value

    return tensor


def laplacian_positional_encoding(adj, k=15, eps=1e-5):
    # Symmetric normalized Laplacian matrix
    degree = adj.sum(1)
    degree_inv_sqrt = torch.pow(degree + eps, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    norm_adj = torch.diag(degree_inv_sqrt) @ adj @ torch.diag(degree_inv_sqrt).to(adj.device)
    L_sym = torch.eye(adj.shape[0]).to(adj.device)
    L_sym = L_sym - norm_adj

    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(L_sym)  # Auto-sorted
    return eigvecs[:, 1:k+1]  # Take eigenvectors corresponding to the k smallest eigenvalues


def visualize_attention_heads(pre_graph_energy, xlogsmask, post_graph_energy, attention, sample_idx=0, save_path=None):
    """
    Display attention matrix comparison of the first 4 heads in one figure, with support for saving vector graphics

    Args:
    - pre_graph_energy: energy matrix before adding graph [batch_size, heads, query_len, key_len]
    - xlogsmask: X-LogSMask matrix of graph structure [batch_size, heads, query_len, key_len]
    - post_graph_energy: energy matrix after adding graph [batch_size, heads, query_len, key_len]
    - attention: final attention matrix [batch_size, heads, query_len, key_len]
    - sample_idx: index of the sample to visualize
    - save_path: vector graphics save path (e.g., "attention_plot.pdf"), None to skip saving
    """
    # Nature figure style: sans-serif text and editable embedded fonts.
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "figure.titlesize": 7,
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.5,
    })

    num_heads = min(4, pre_graph_energy.shape[1])  # Show at most 4 heads

    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad(color="white", alpha=1.0)

    # Create subplots close to Nature double-column width (~183 mm wide for 4 heads)
    fig_width = max(89 / 25.4, num_heads * 183 / 25.4 / 4)
    fig, axes = plt.subplots(3, num_heads, figsize=(fig_width, 5.4), squeeze=False)
    fig.patch.set_facecolor("white")

    # Set overall title
    # fig.suptitle('Energy Matrix with/without X-LogSMask', fontsize=16)

    # Iterate over first 4 heads
    for head_idx in range(num_heads):
        # Get data for specific head and sample
        pre_energy_head = pre_graph_energy[sample_idx, head_idx].cpu().numpy()
        xlogsmask_head = xlogsmask[sample_idx, head_idx].cpu().numpy()
        post_energy_head = post_graph_energy[sample_idx, head_idx].cpu().numpy()
        pre_energy_head = np.ma.masked_invalid(pre_energy_head)
        xlogsmask_head = np.ma.masked_invalid(xlogsmask_head)
        post_energy_head = np.ma.masked_invalid(post_energy_head)

        # Draw energy matrix before adding graph (first row)
        axes[0, head_idx].set_facecolor("white")
        im1 = axes[0, head_idx].imshow(pre_energy_head, cmap=cmap, interpolation="nearest", aspect="equal")
        axes[0, head_idx].set_title(f'Head {head_idx+1}')
        if head_idx == 0:
            axes[0, head_idx].set_ylabel("Energy")
        axes[0, head_idx].set_xticks([])
        axes[0, head_idx].set_yticks([])
        cbar1 = plt.colorbar(im1, ax=axes[0, head_idx], shrink=0.65)
        cbar1.ax.tick_params(labelsize=5, width=0.5, length=2, pad=1)
        cbar1.outline.set_linewidth(0.5)

        # Draw X-LogSMask matrix (second row)
        axes[1, head_idx].set_facecolor("white")
        im2 = axes[1, head_idx].imshow(xlogsmask_head, cmap=cmap, interpolation="nearest", aspect="equal")
        if head_idx == 0:
            axes[1, head_idx].set_ylabel("X-LogSMask")
        axes[1, head_idx].set_xticks([])
        axes[1, head_idx].set_yticks([])
        cbar2 = plt.colorbar(im2, ax=axes[1, head_idx], shrink=0.65)
        cbar2.ax.tick_params(labelsize=5, width=0.5, length=2, pad=1)
        cbar2.outline.set_linewidth(0.5)

        # Draw energy matrix after adding graph (third row)
        axes[2, head_idx].set_facecolor("white")
        im3 = axes[2, head_idx].imshow(post_energy_head, cmap=cmap, interpolation="nearest", aspect="equal")
        if head_idx == 0:
            axes[2, head_idx].set_ylabel("Energy + X-LogSMask")
        axes[2, head_idx].set_xticks([])
        axes[2, head_idx].set_yticks([])
        cbar3 = plt.colorbar(im3, ax=axes[2, head_idx], shrink=0.65)
        cbar3.ax.tick_params(labelsize=5, width=0.5, length=2, pad=1)
        cbar3.outline.set_linewidth(0.5)

    plt.tight_layout()

    # Save vector graphics
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            create_dir(save_dir)
        _, extension = os.path.splitext(save_path)
        save_format = extension[1:].lower() if extension else None
        plt.savefig(save_path, format=save_format, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='white', transparent=False)
        print(f"Figure saved to: {save_path}")

    plt.show()
