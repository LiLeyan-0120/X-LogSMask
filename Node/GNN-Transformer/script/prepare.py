"""Data preprocessing

    1. Normalize node features
    2. Partition nodes into training set, validation set and test set
    3. Normalize adjacency matrix
    4. Load data to corresponding device, cpu or gpu

"""


import torch
import scipy
import numpy as np
from .utils import PrepData, laplacian_positional_encoding
from scipy.sparse import csr_matrix


def normalize_adjacency(adjacency):
    """Adjacency matrix normalization

        L = D^-0.5 * (A + I) * D^-0.5
        A: adjacency matrix, L: normalized adjacency matrix

        Input:
        ------
        adjacency: sparse numpy array, adjacency matrix

        Output:
        -------
        norm_adjacency: sparse numpy array, normalized adjacency matrix

    """

    adjacency += scipy.sparse.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = np.power(degree, -0.5).flatten()
    d_hat[np.isinf(d_hat)] = 0.0
    d_hat = scipy.sparse.diags(d_hat)
    norm_adjacency = d_hat.dot(adjacency).dot(d_hat).tocoo()

    return norm_adjacency


def prepare(dataset):
    """Data preprocessing

        1. Normalize node features
        2. Partition nodes into training set, validation set and test set
        3. Normalize adjacency matrix
        4. Load data to corresponding device, cpu or gpu

        Input:
        ------
        dataset: Data, containing elements:
                 X: numpy array, node features
                 y: numpy array, node class labels
                 adjacency: sparse numpy array, adjacency matrix
                 test_mask: numpy array, test set sample mask
                 train_mask: numpy array, training set sample mask
                 valid_mask: numpy array, validation set sample mask

        Output:
        -------
        dataset: Data, containing elements:
                 X: tensor, normalized node features
                 y: tensor, node class labels
                 adjacency: tensor, normalized adjacency matrix
                 test_index: tensor, test set sample indices
                 train_index: tensor, training set sample indices
                 valid_index: tensor, validation set sample indices

    """

    # Node feature normalization
    X_rowsum = dataset.data.X.sum(1, keepdims=True)
    X_rowsum[X_rowsum == 0] = 1
    X = dataset.data.X / X_rowsum
    X = torch.FloatTensor(X)

    # Node labels
    y = torch.LongTensor(dataset.data.y)

    # Sample indices for each data split
    test_index = torch.LongTensor(np.where(dataset.data.test_mask)[0])
    train_index = torch.LongTensor(np.where(dataset.data.train_mask)[0])
    valid_index = torch.LongTensor(np.where(dataset.data.valid_mask)[0])

    # Adjacency matrix normalization
    norm_adjacency = normalize_adjacency(dataset.data.adjacency)  # Mitigate the influence of high-degree nodes # Also very important for high head counts, as multi-step connections tend toward poor patterns
    # norm_adjacency = dataset.data.adjacency
    edges = np.asarray([norm_adjacency.row, norm_adjacency.col])
    edges = torch.from_numpy(edges.astype(int)).long()
    values = torch.from_numpy(norm_adjacency.data.astype(np.float32))
    adjacency = torch.sparse_coo_tensor(edges, values, (len(X), len(X)), dtype=torch.float32).to_dense()
    # pe = laplacian_positional_encoding(adjacency, k=5)

    # Target device for data loading
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Merge data
    dataset = PrepData(
        X=X.to(device),
        y=y.to(device),
        edges=edges.to(device),
        adjacency=adjacency.to(device),
        test_index=test_index.to(device),
        train_index=train_index.to(device),
        valid_index=valid_index.to(device)
        # pe=pe.to(device)
    )

    return dataset
