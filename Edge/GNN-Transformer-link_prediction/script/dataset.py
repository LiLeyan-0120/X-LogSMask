import math
import warnings
from torch_geometric.datasets import Planetoid
from ogb.linkproppred import PygLinkPropPredDataset
import torch
from torch_geometric.utils import degree
import torch_geometric.transforms as T

# Filter FutureWarning warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class LinkPredDataset(object):
    """Load graph data
    """

    def __init__(self, data, dataset_root, **params):
        """
        Inputs:
            data: Dataset name, ['Cora', 'Citeseer', 'Pubmed', 'Collab', 'PPA', 'Citation2', 'DDI']
            dataset_root: Dataset save path
            params: Configuration parameters, must include split (valid_prop/test_prop)
        """
        assert data in ['cora', 'citeseer', 'pubmed', 'collab', 'PPA', 'Citation2', 'DDI'], f'Unknown dataset: {data}'
        self.data_name = data
        self.dataset_root = dataset_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = None
        self.all_pos_edges = None
        self.split_edge = None

        # Load corresponding dataset
        print(f'[Data Loading] Processing {self.data_name.upper()} dataset...')
        if self.data_name in ['cora', 'citeseer', 'pubmed']:
            self._load_planetoid(**params)
        else:
            self._load_ogb()

        if self.dataset.x is None:
            feature_type = 'random_emb'
            if feature_type in ['degree_onehot', 'degree_norm']:
                # Original degree-based processing logic
                deg = degree(self.dataset.edge_index[0], dtype=torch.long)
                max_degree = deg.max().item()
                if feature_type == 'degree_onehot':
                    transform = T.OneHotDegree(max_degree)
                else:
                    deg = deg.to(torch.float)
                    mean, std = deg.mean().item(), deg.std().item()
                    transform = NormalizedDegree(mean, std)
            elif feature_type == 'random_emb':
                # Use random learnable embeddings
                init_type = 'orthogonal'  # Can be 'normal', 'uniform', 'orthogonal'
                learnable = True  # Set to True to create learnable embedding layer, False for fixed features
                transform = RandomNodeEmbedding(emb_dim=128, init_type=init_type, learnable=learnable)  # Pass embedding dimension, initialization type and learnability
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            self.dataset = transform(self.dataset)

    def _load_planetoid(self, **params):
        """Load Planetoid dataset, supports custom positive/negative sample ratios"""
        # 1. Load raw dataset
        raw_dataset = Planetoid(root=self.dataset_root, name=self.data_name)
        self.dataset = raw_dataset[0]
        self.dataset.x = self.dataset.x if self.dataset.x is not None else None
        num_nodes = self.dataset.num_nodes
        row, col = self.dataset.edge_index

        # 2. Undirected edge deduplication
        mask_undir = row < col
        undir_row = row[mask_undir]
        undir_col = col[mask_undir]
        undir_pos_edges = torch.stack([undir_row, undir_col], dim=0)
        total_undir_pos = undir_pos_edges.size(1)

        # 3. Split positive edges
        valid_ratio = params['split']['valid_prop']
        test_ratio = params['split']['test_prop']
        n_v = int(math.floor(valid_ratio * total_undir_pos))  # Number of validation positive edges
        n_t = int(math.floor(test_ratio * total_undir_pos))  # Number of test positive edges

        perm = torch.randperm(total_undir_pos)
        shuffled_undir_row = undir_row[perm]
        shuffled_undir_col = undir_col[perm]

        val_undir_row, val_undir_col = shuffled_undir_row[:n_v], shuffled_undir_col[:n_v]
        test_undir_row, test_undir_col = shuffled_undir_row[n_v:n_v + n_t], shuffled_undir_col[n_v:n_v + n_t]
        train_undir_row, train_undir_col = shuffled_undir_row[n_v + n_t:], shuffled_undir_col[n_v + n_t:]

        # Convert to bidirectional edges
        def undir_to_bidir(undir_row, undir_col):
            bidir_row = torch.cat([undir_row, undir_col], dim=0)
            bidir_col = torch.cat([undir_col, undir_row], dim=0)
            return torch.stack([bidir_row, bidir_col], dim=0)

        self.dataset.train_pos_edge_index = undir_to_bidir(train_undir_row, train_undir_col)
        self.dataset.valid_pos_edge_index = undir_to_bidir(val_undir_row, val_undir_col)
        self.dataset.test_pos_edge_index = undir_to_bidir(test_undir_row, test_undir_col)

        # 4. Generate negative edges (core modification: support custom positive/negative ratios)
        # 4.1 Get custom negative sample ratio
        neg_ratio = params['split'].get('neg_ratio', 1)  # e.g., neg_ratio=5 means 1:5
        # Calculate number of negative edges needed for validation and test sets (positive edge count x negative sample ratio)
        val_neg_num = int(n_v * neg_ratio)  # Validation negative edges = validation positive edges x neg_ratio
        test_neg_num = int(n_t * neg_ratio)  # Test negative edges = test positive edges x neg_ratio
        total_neg_needed = val_neg_num + test_neg_num  # Total negative edges to sample

        # 4.2 Build negative edge candidate mask
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)  # Keep upper triangle where u < v, exclude self-loops
        neg_adj_mask[undir_row, undir_col] = False  # Exclude real positive edges

        # 4.3 Sample negative edges
        neg_candidates = neg_adj_mask.nonzero(as_tuple=False).t()  # All possible negative edge candidates
        available_neg = neg_candidates.size(1)  # Total candidate negative edges
        # Ensure sampling count does not exceed total candidates
        total_neg_sampled = min(total_neg_needed, available_neg)
        if total_neg_sampled < total_neg_needed:
            print(f"Warning: Insufficient candidate negative edges, sampled {total_neg_sampled} (requested {total_neg_needed})")

        # Randomly sample specified number of negative edges
        perm_neg = torch.randperm(available_neg)[:total_neg_sampled]
        sampled_neg_row, sampled_neg_col = neg_candidates[0][perm_neg], neg_candidates[1][perm_neg]

        # 4.4 Split validation and test negative edges
        # Validation negative edges: first val_neg_num entries
        val_neg_undir_row, val_neg_undir_col = sampled_neg_row[:val_neg_num], sampled_neg_col[:val_neg_num]
        # Test negative edges: remaining test_neg_num entries
        test_neg_undir_row, test_neg_undir_col = sampled_neg_row[
                                                 val_neg_num:val_neg_num + test_neg_num], sampled_neg_col[
                                                                                          val_neg_num:val_neg_num + test_neg_num]

        # Convert negative edges to bidirectional
        self.dataset.valid_neg_edge_index = undir_to_bidir(val_neg_undir_row, val_neg_undir_col)
        self.dataset.test_neg_edge_index = undir_to_bidir(test_neg_undir_row, test_neg_undir_col)

        # 4.5 Training negative edge mask (exclude sampled validation/test negative edges)
        neg_adj_mask[sampled_neg_row, sampled_neg_col] = False
        self.dataset.train_neg_adj_mask = neg_adj_mask


    def _load_ogb(self):
        """Load OGB dataset (Collab/PPA/Citation2/DDI)"""
        # 1. Load OGB raw data
        ogb_dataset = PygLinkPropPredDataset(
            name=f'ogbl-{self.data_name.lower()}',
            root=self.dataset_root
        )
        self.dataset = ogb_dataset[0].to(self.device)
        self.split_edge = ogb_dataset.get_edge_split()

        if self.data_name != 'Citation2':
            self.dataset.train_pos_edge_index = self.split_edge['train']['edge'].t()

            # Validation set
            self.dataset.valid_pos_edge_index = self.split_edge['valid']['edge'].t()
            self.dataset.valid_neg_edge_index = self.split_edge['valid']['edge_neg'].t()  # OGB provides validation negative samples

            # Test set
            self.dataset.test_pos_edge_index = self.split_edge['test']['edge'].t()
            self.dataset.test_neg_edge_index = self.split_edge['test']['edge_neg'].t()  # OGB provides test negative samples
        else:
            # Citation2 special format: split_edge[split]['source_node']/['target_node'] are (N,) numpy arrays
            self.dataset.train_pos_edge_index = torch.stack([self.split_edge['train']['source_node'], self.split_edge['train']['target_node']], dim=0)
            self.dataset.valid_pos_edge_index = torch.stack([self.split_edge['valid']['source_node'], self.split_edge['valid']['target_node']], dim=0)
            valid_neg_node = self.split_edge['valid']['target_node_neg']
            self.dataset.valid_neg_edge_index = self._format_citation2_neg(self.split_edge['valid']['source_node'], valid_neg_node)
            self.dataset.test_pos_edge_index = torch.stack([self.split_edge['test']['source_node'], self.split_edge['test']['target_node']], dim=0)
            test_neg_node = self.split_edge['test']['target_node_neg']
            self.dataset.test_neg_edge_index = self._format_citation2_neg(self.split_edge['test']['source_node'],
                                                                           test_neg_node)

    def _format_citation2_neg(self, source_nodes, neg_targets):
        """Convert Citation2 negative sample format to (2, num_neg_edges)

        Args:
            source_nodes: Source node array, shape=[num_pos_edges]
            neg_targets: Negative sample target node array, shape=[num_pos_edges, 1000]
        Returns:
            Negative edge index, shape=[2, num_pos_edges*1000]
        """
        k = neg_targets.size(1)  # 1000, number of negative samples per positive edge

        # Expand source nodes: each source node repeats k times (corresponding to negative sample target nodes)
        expanded_source = source_nodes.unsqueeze(1).repeat(1, k).flatten()  # shape: [num_pos*1000]

        # Flatten negative sample target nodes
        flattened_target = neg_targets.flatten()  # shape: [num_pos*1000]

        # Combine into edge index format
        return torch.stack([expanded_source, flattened_target], dim=0)


class GlobalUniform:
    def __init__(self, num_neg):
        """
        Initialize global uniform negative sampler
        :param num_neg: Number of negative samples to sample per source node
        """
        self.num_neg = num_neg

    def __call__(self, num_nodes, pos_edges, sources):
        """
        Sample negative edges for a list of source nodes
        :param num_nodes: Total number of nodes in the graph (used to limit sampling range)
        :param pos_edges: Index of all positive edges in the graph, shape [2, E] (first row is source nodes, second row is target nodes)
        :param sources: List of source nodes that need negative edge sampling, shape [N] (N is the number of positive edges)
        :return: List of target nodes for negative edges, shape [N, num_neg] (each row corresponds to negative target nodes for a source node in sources)
        """
        # 1. Build positive edge set (quickly determine if (u, v) is a positive edge)
        # Convert positive edges to tuple set, e.g., {(u1, v1), (u2, v2), ...}
        pos_set = set()
        u_list = pos_edges[0].tolist()  # Source node list of positive edges
        v_list = pos_edges[1].tolist()  # Target node list of positive edges
        for u, v in zip(u_list, v_list):
            pos_set.add((u, v))  # If the graph is undirected, pos_edges already contains bidirectional edges, no need to add (v, u) again

        # 2. Sample negative edges for each source node
        neg_targets = []  # Store negative target nodes for each source node
        for u in sources:
            u = u.item()  # Convert to scalar (avoid tensor operations)
            current_neg = []  # Negative target nodes for current source node

            # Sample until num_neg valid negative edges are obtained
            while len(current_neg) < self.num_neg:
                # Generate candidate target nodes in batch (2x the current need for efficiency)
                need = self.num_neg - len(current_neg)
                candidates = torch.randint(0, num_nodes, (need * 2,), dtype=torch.long).tolist()

                # Filter out candidate nodes that belong to positive edges
                valid_candidates = [v for v in candidates if (u, v) not in pos_set]

                # Take enough valid candidate nodes
                current_neg.extend(valid_candidates[:need])

            neg_targets.append(current_neg)

        # Convert to tensor, shape [N, num_neg]
        return torch.tensor(neg_targets, dtype=torch.long)

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class RandomNodeEmbedding(T.BaseTransform):
    def __init__(self, emb_dim, init_type='normal', learnable=False):
        """
        Initialize random node embedding transform
        Args:
            emb_dim: Embedding dimension (hyperparameter)
            init_type: Initialization type, options are 'normal', 'uniform', 'ones', 'orthogonal'
            learnable: Whether to create a learnable embedding layer, if True creates torch.nn.Embedding
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
            data.x = embedding.weight
        else:
            # Create non-learnable fixed features
            if self.init_type == 'normal':
                data.x = torch.randn(num_nodes, self.emb_dim)  # Random normal distribution initialization
            elif self.init_type == 'uniform':
                data.x = torch.rand(num_nodes, self.emb_dim)  # Uniform distribution initialization
            elif self.init_type == 'orthogonal':
                # Orthogonal initialization
                # Create a tensor with shape (num_nodes, emb_dim)
                data.x = torch.empty(num_nodes, self.emb_dim)
                # Use orthogonal initialization
                torch.nn.init.orthogonal_(data.x)
            else:
                raise ValueError(f"Unknown initialization type: {self.init_type}")

        return data
