import torch
import numpy as np

def image_to_graph(images, patch_size):
    """Convert images to graph structure

    Args:
        images (torch.Tensor): Batch image tensor [B, C, H, W]
        patch_size (int): Patch size

    Returns:
        patches (torch.Tensor): Image patches [B, N, C*P*P]
        adjacency (torch.Tensor): Adjacency matrix [B, N, N]
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"

    # Compute number of patches
    num_patches = (H // patch_size) * (W // patch_size)

    # Split image into patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # [B, N, C, P, P]
    patches = patches.view(B, num_patches, -1)  # [B, N, C*P*P]

    # Create adjacency matrix
    grid_size = H // patch_size  # Grid size
    adjacency = torch.zeros(B, num_patches, num_patches).to(patches.device)

    # Compute coordinates for each patch
    patch_coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch_coords.append((i, j))

    # Build adjacency matrix - connect adjacent patches
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:
                adjacency[:, i, j] = 1

            # Get patch coordinates
            i_row, i_col = patch_coords[i]
            j_row, j_col = patch_coords[j]

            # Check if adjacent (including diagonal)
            if abs(i_row - j_row) <= 1 and abs(i_col - j_col) <= 1:
                adjacency[:, i, j] = 1

    return patches, adjacency

def compute_normal_adjacency(batch_adjacency):
    """Batch higher-order adjacency matrix normalization"""

    # Compute degree matrix D (maintaining batch dimension)
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

def multi_graph(graph, num_heads):
    """Create multi-graph structure for multi-head attention

    Args:
        graph (torch.Tensor): Original adjacency matrix [B, N, N]
        num_heads (int): Number of attention heads

    Returns:
        multigraphs (torch.Tensor): Multi-graph structure [B, num_heads, N, N]
    """
    # Initialize tensor list and add original adjacency matrix
    graph_list = [graph]

    # Compute matrix powers sequentially
    for power in range(2, num_heads+1):
        graph_n = torch.matmul(graph_list[-1], graph)
        graph_list.append(graph_n)

    # Stack all tensors together to form a tensor with shape (B, num_heads, N, N)
    multigraphs = torch.stack(graph_list, dim=1)

    return multigraphs


def multi_graph_extra(multigraph):
    """Add extra rows and columns (all ones) to multi-graph structure

    Args:
        multigraph (torch.Tensor): Input multi-graph tensor [B, H, N, N]

    Returns:
        multigraph_extra (torch.Tensor): Multi-graph tensor with extra rows and columns [B, H, N+1, N+1]
    """
    batch_size, num_heads, n_nodes, _ = multigraph.shape
    device = multigraph.device
    
    # Create a zero matrix with shape [B, H, N+1, N+1]
    multigraph_extra = torch.zeros(batch_size, num_heads, n_nodes + 1, n_nodes + 1, device=device)
    
    # Copy original multi-graph to the top-left corner of the new matrix
    multigraph_extra[:, :, :n_nodes, :n_nodes] = multigraph
    
    # Set the last row and last column to 1
    multigraph_extra[:, :, -1, :] = 1
    multigraph_extra[:, :, :, -1] = 1
    
    return multigraph_extra
