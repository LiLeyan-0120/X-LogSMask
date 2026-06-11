
import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class Dataset(object):
    """Load edge regression dataset"""

    def __init__(self, data, dataset_root, **params):
        """
        Args:
            data: dataset name, supports ['epic-games-plr', 'air-traffic-2019-rlr', 'air-traffic-2015-rlr']
            dataset_root: dataset save path
            params: configuration parameters, must include split (valid_prop/test_prop)
        """
        assert data in ['epic-games-plr', 'air-traffic-2019-rlr', 'air-traffic-2015-rlr'], f'Unknown dataset: {data}'
        self.data_name = data
        self.dataset_root = dataset_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = None

        # Load corresponding dataset
        print(f'[Data Loading] Processing {self.data_name} dataset...')
        self._load_edge_regression_dataset(**params)

        # If no node features exist, generate random embeddings
        if 'node_attr' not in self.dataset or self.dataset['node_attr'] is None:
            feature_type = 'random_emb'
            if feature_type == 'random_emb':
                init_type = 'orthogonal'
                learnable = True
                transform = RandomNodeEmbedding(emb_dim=128, init_type=init_type, learnable=learnable)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            # Convert dict to Data object for the transform
            data_obj = Data()
            for key, value in self.dataset.items():
                setattr(data_obj, key, value)
            data_obj = transform(data_obj)
            self.dataset = data_obj
        else:
            # Convert dictionary to PyTorch Geometric Data object for compatibility with pipeline
            data_obj = Data()
            for key, value in self.dataset.items():
                setattr(data_obj, key, value)
            self.dataset = data_obj

    def _load_edge_regression_dataset(self, **params):
        """Load edge regression dataset"""
        # Build dataset file path
        data_file = os.path.join(self.dataset_root, f"{self.data_name}.pt")

        # Check if dataset file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file does not exist: {data_file}")

        # Load .pt file directly
        data = torch.load(data_file)

        # Use loaded data
        self.dataset = data

        # Split into training, validation, and test sets
        self._split_edges(**params)

    def _split_edges(self, **params):
        """Split edge dataset"""
        # Get all edge indices and weights
        edge_index = self.dataset['edge_index']
        edge_attr = self.dataset['edge_attr']
        
        # Get undirected edges (deduplicate)
        mask_undir = edge_index[0] < edge_index[1]
        undir_edge_index = edge_index[:, mask_undir]
        undir_edge_weights = edge_attr[mask_undir]

        # Randomly shuffle edges
        num_edges = undir_edge_index.size(1)
        perm = torch.randperm(num_edges)
        undir_edge_index = undir_edge_index[:, perm]
        undir_edge_weights = undir_edge_weights[perm]

        # Calculate split sizes
        valid_ratio = params['split']['valid_prop']
        test_ratio = params['split']['test_prop']
        n_v = int(valid_ratio * num_edges)
        n_t = int(test_ratio * num_edges)

        # Split edges
        val_edge_index = undir_edge_index[:, :n_v]
        val_edge_weights = undir_edge_weights[:n_v]

        test_edge_index = undir_edge_index[:, n_v:n_v+n_t]
        test_edge_weights = undir_edge_weights[n_v:n_v+n_t]

        train_edge_index = undir_edge_index[:, n_v+n_t:]
        train_edge_weights = undir_edge_weights[n_v+n_t:]

        # Convert to bidirectional edges
        def to_bidir(edge_idx, edge_w):
            # Add reverse edges
            reverse_idx = torch.stack([edge_idx[1], edge_idx[0]], dim=0)
            bidir_idx = torch.cat([edge_idx, reverse_idx], dim=1)
            # Copy weights, ensure correct shape
            if edge_w.dim() == 1:
                # If 1D tensor, copy directly
                bidir_w = torch.cat([edge_w, edge_w], dim=0)
            else:
                # If multi-dimensional tensor, maintain shape
                bidir_w = torch.cat([edge_w, edge_w], dim=0)
            return bidir_idx, bidir_w

        # Save split results (using dictionary access since self.dataset is a dict)
        self.dataset['train_edge_index'], self.dataset['train_edge_attr'] = to_bidir(train_edge_index, train_edge_weights)
        self.dataset['valid_edge_index'], self.dataset['valid_edge_attr'] = to_bidir(val_edge_index, val_edge_weights)
        self.dataset['test_edge_index'], self.dataset['test_edge_attr'] = to_bidir(test_edge_index, test_edge_weights)


class RandomNodeEmbedding(T.BaseTransform):
    def __init__(self, emb_dim, init_type='normal', learnable=False):
        """
        Initialize random node embedding transform
        Args:
            emb_dim: embedding dimension (hyperparameter)
            init_type: initialization type, options are 'normal', 'uniform', 'ones', 'orthogonal'
            learnable: whether to create a learnable embedding layer, if True creates torch.nn.Embedding
        """
        self.emb_dim = emb_dim
        self.init_type = init_type
        self.learnable = learnable

    def __call__(self, data):
        # Generate random embedding vector for each node (shape: [num_nodes, emb_dim])
        num_nodes = data.num_nodes

        if self.learnable:
            # Create learnable embedding layer
            embedding = torch.nn.Embedding(num_nodes, self.emb_dim)

            # Initialize weights based on initialization type
            if self.init_type == 'normal':
                torch.nn.init.normal_(embedding.weight)
            elif self.init_type == 'uniform':
                torch.nn.init.uniform_(embedding.weight)
            elif self.init_type == 'orthogonal':
                torch.nn.init.orthogonal_(embedding.weight)
            else:
                raise ValueError(f"Unknown initialization type: {self.init_type}")

            # Save embedding layer for later use
            data.embedding_layer = embedding
            # Use current weights as initial features
            data.node_attr = embedding.weight
        else:
            # Create non-learnable fixed features
            if self.init_type == 'normal':
                data.node_attr = torch.randn(num_nodes, self.emb_dim)  # Random normal distribution initialization
            elif self.init_type == 'uniform':
                data.node_attr = torch.rand(num_nodes, self.emb_dim)  # Uniform distribution initialization
            elif self.init_type == 'orthogonal':
                # Orthogonal initialization
                # Create a tensor with shape (num_nodes, emb_dim)
                data.node_attr = torch.empty(num_nodes, self.emb_dim)
                # Use orthogonal initialization
                torch.nn.init.orthogonal_(data.node_attr)
            else:
                raise ValueError(f"Unknown initialization type: {self.init_type}")

        return data
