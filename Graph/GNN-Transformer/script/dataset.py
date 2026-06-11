
import random
from torch_geometric.datasets import TUDataset, ZINC
import torch
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset
import warnings
warnings.filterwarnings(
    "ignore",
    message="It is not recommended to directly access the internal storage format `data` of an 'InMemoryDataset'.*",
    category=UserWarning,
    module="torch_geometric.data.in_memory_dataset"
)

class Dataset(object):
    """Load graph data

        Download and preprocess related datasets using TUDataset

    """

    def __init__(self, data, dataset_root, **params):
        """Load graph data

            Preprocessed dataset save path: $dataset_root/$data

            Inputs:
            -------
            data: string, name of the dataset to use
            dataset_root: string, root folder path for saving datasets
            params: dict, contains split sub-dictionary providing validation and test set data ratios

        """

        assert data in ['DD', 'NCI1', 'PROTEINS', 'MUTAG', 'COLLAB', 'IMDB-BINARY', 'ZINC_full', 'ZINC', 'MOLHIV'], 'uknown dataset'

        # Download and preprocess related datasets
        print('Downloading and Preprocessing [{}] Dataset ...'.format(data.upper()))
        if data == 'ZINC':
            self.train = ZINC(root=dataset_root+'/ZINC', subset=True, split='train')
            self.valid = ZINC(root=dataset_root+'/ZINC', subset=True, split='val')
            self.test = ZINC(root=dataset_root+'/ZINC', subset=True, split='test')
        elif data == 'ZINC_full':
            self.train = ZINC(root=dataset_root + '/ZINC_full', subset=False, split='train')
            self.valid = ZINC(root=dataset_root + '/ZINC_full', subset=False, split='val')
            self.test = ZINC(root=dataset_root + '/ZINC_full', subset=False, split='test')
        elif data == 'MOLHIV':
            self.dataset = PygGraphPropPredDataset(
                name='ogbg-molhiv',  # OGB dataset name
                root=dataset_root  # Storage path
            )
            # Get OGB predefined train/valid/test split indices
            split_idx = self.dataset.get_idx_split()
            # Extract subsets by indices
            self.train = self.dataset[split_idx['train']]
            self.valid = self.dataset[split_idx['valid']]
            self.test = self.dataset[split_idx['test']]
        else:
            # Original TUDataset loading method
            self.dataset = TUDataset(root=dataset_root, name=data, use_node_attr=True)
            if self.dataset.data.x is None:
                max_degree = 0
                degs = []
                for data in self.dataset:
                    degs += [degree(data.edge_index[0], dtype=torch.long)]
                    max_degree = max(max_degree, degs[-1].max().item())
                if max_degree < 1000:
                    self.dataset.transform = T.OneHotDegree(max_degree)
                else:
                    deg = torch.cat(degs, dim=0).to(torch.float)
                    mean, std = deg.mean().item(), deg.std().item()
                    self.dataset.transform = NormalizedDegree(mean, std)

            # Dataset split
            self.__split_data(**params)

        return

    def __split_data(self, **params):
        """Dataset split

            Split the dataset into training, validation, and test sets

            Input:
            ------
            params: dict, contains split sub-dictionary providing validation and test set data ratios

        """

        # Number of graphs
        num_graphs = len(self.dataset)

        # Shuffle initial graph sample indices
        indices = list(range(num_graphs))
        random.seed(params['random_state'])
        random.shuffle(indices)

        # Number of test and validation graphs
        num_test = int(num_graphs * params['split']['test_prop'])
        num_valid = int(num_graphs * params['split']['valid_prop'])

        # Get test, validation, and training set graph sample indices
        test_indices = indices[:num_test]
        valid_indices = indices[num_test:num_test + num_valid]
        train_indices = indices[num_test + num_valid:]

        # Get test, validation, and training set graph samples
        self.test = self.dataset[test_indices]
        self.valid = self.dataset[valid_indices]
        self.train = self.dataset[train_indices]

        return

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
