"""Download, storage and preprocessing of datasets

    Download, storage and preprocessing of raw datasets,
    Raw dataset storage path: $dataset_root/$data

"""


import os
import pickle
import numpy as np
import urllib.request
from torch_geometric.datasets import Amazon, Coauthor, WikiCS
import torch_geometric.transforms as T
from .utils import *
from itertools import groupby
from scipy.sparse import coo_matrix


class Dataset(object):
    """Download, storage and preprocessing of datasets

        Download, storage and preprocessing of raw datasets,
        Raw dataset storage path: $dataset_root/$data

    """

    url_root = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    files = [
        'ind.{}.x', 'ind.{}.tx', 'ind.{}.allx',
        'ind.{}.y', 'ind.{}.ty', 'ind.{}.ally',
        'ind.{}.graph', 'ind.{}.test.index'
    ]

    def __init__(self, data, dataset_root):
        """Dataset initialization

            Download, storage and preprocessing of raw datasets,
            Raw dataset storage path: $dataset_root/$data

            Inputs:
            -------
            data: string, dataset name, ['cora', 'pubmed', 'citeseer']
            dataset_root: string, root folder path for saving datasets

        """

        assert data in ['cora', 'pubmed', 'citeseer', 'Computers', 'Photo', 'CS', 'Physics', 'WikiCS'], 'uknown dataset'

        self.data = data
        self.dataset_root = dataset_root
        self.data_dir = os.path.join(dataset_root, data)

        # Download and preprocess dataset if it does not exist or update existing dataset
        print('Downloading and Preprocessing [{}] Dataset ...'.format(data.upper()))
        self.__download_data()
        if self.data == 'cora' or self.data == 'citeseer' or self.data == 'pubmed':
            self.data = self.__process_data()
        else:
            self.data = self.__process_data1()
        return

    # ------------------------------------------------------------------------
    # Download dataset

    def __download_data(self):
        """Download raw dataset

            Download each data file separately and save to self.data_dir folder

        """

        # Create self.data_dir folder
        create_dir(self.data_dir)

        if self.data == 'cora' or self.data == 'citeseer' or self.data == 'pubmed':
            for name in self.files:
                # Iterate over each data file
                name = name.format(self.data)
                file = os.path.join(self.data_dir, name)
                if not os.path.isfile(file):
                    # Download file if it does not exist
                    url = '{}/{}'.format(self.url_root, name)
                    self.__download_from_url(url, file)
        elif self.data == 'Photo' or self.data == 'Computers':
            self.dataset = Amazon(root=self.dataset_root, name=self.data, transform=T.NormalizeFeatures())
        elif self.data == 'CS' or self.data == 'Physics':
            self.dataset = Coauthor(root=self.dataset_root, name=self.data, transform=T.NormalizeFeatures())
        elif self.data == 'WikiCS':
            self.dataset = WikiCS(root=self.dataset_root+'/WikiCS', transform=T.NormalizeFeatures(), is_undirected=False)
        return

    def __download_from_url(self, url, file):
        """Download file from URL

            Download file from url and save to file

            Inputs:
            -------
            url: string, file download link
            file: string, download file save path

        """

        try:
            # Establish file connection and write file
            data = urllib.request.urlopen(url, timeout=100)
            with open(file, 'wb') as f:
                f.write(data.read())
            data.close()
        except Exception:
            # Download failed, retry download
            self.__download_from_url(url, file)

        return

    # ------------------------------------------------------------------------
    # Preprocess dataset

    def __process_data(self):
        """Data processing

            Process data to obtain node features and labels, adjacency matrix,
            training set, validation set and test set

            Output:
            -------
            dataset: Data tuple, preprocessed dataset, containing elements:
                     X: numpy array, node features
                     y: numpy array, node class labels
                     adjacency: sparse numpy array, adjacency matrix
                     test_mask: numpy array, test set sample mask
                     train_mask: numpy array, training set sample mask
                     valid_mask: numpy array, validation set sample mask

        """

        # Read data
        _, tx, allx, y, ty, ally, graph, test_index = \
            [self.read_data(os.path.join(self.data_dir, file.format(self.data))) for file in self.files]

        # Training set and validation set sample indices
        train_index = np.arange(len(y))
        valid_index = np.arange(len(y), len(y) + 500)
        # Use full split for training set
        # train_index = np.arange(len(ally) - 500)
        # valid_index = np.arange(len(ally) - 500, len(ally))

        # Merge other samples with test set samples
        X = np.concatenate([allx, tx], axis=0)
        y = np.concatenate([ally, ty], axis=0).argmax(axis=1)

        # Build adjacency matrix based on graph structure
        adjacency = self.build_adjacency(graph)

        if self.data == 'citeseer':
            # Special processing for citeseer dataset
            num_nodes = adjacency.shape[0]
            test_index, adjacency = self.__citeseer(tx, test_index, graph, num_nodes)

        # Sort test set samples in order
        sorted_test_index = sorted(test_index)
        X[test_index] = X[sorted_test_index]
        y[test_index] = y[sorted_test_index]

        # Training set, validation set and test set node partitioning
        num_nodes = len(X)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask = np.zeros(num_nodes, dtype=bool)
        valid_mask = np.zeros(num_nodes, dtype=bool)
        test_mask[test_index] = True
        train_mask[train_index] = True
        valid_mask[valid_index] = True

        # Merge data
        dataset = Data(
            X=X,
            y=y,
            adjacency=adjacency,
            test_mask=test_mask,
            train_mask=train_mask,
            valid_mask=valid_mask
        )

        return dataset

    def __process_data1(self):
        """Data processing

            Process data to obtain node features and labels, adjacency matrix,
            training set, validation set and test set

            Output:
            -------
            dataset: Data tuple, preprocessed dataset, containing elements:
                     X: numpy array, node features
                     y: numpy array, node class labels
                     adjacency: sparse numpy array, adjacency matrix
                     test_mask: numpy array, test set sample mask
                     train_mask: numpy array, training set sample mask
                     valid_mask: numpy array, validation set sample mask

        """

        # Read data
        data = self.dataset[0]

        # Training set and validation set sample indices
        index = data.num_nodes
        train_index = np.arange(int(index*0.8))
        valid_index = np.arange(int(index*0.8), int(index*0.9))
        test_index = np.arange(int(index*0.9), index)

        # Merge other samples with test set samples
        X = data.x
        y = data.y

        # Build adjacency matrix based on graph structure
        adjacency = self.build_adjacency1(data)

        # Training set, validation set and test set node partitioning
        test_mask = np.zeros(index, dtype=bool)
        train_mask = np.zeros(index, dtype=bool)
        valid_mask = np.zeros(index, dtype=bool)
        test_mask[test_index] = True
        train_mask[train_index] = True
        valid_mask[valid_index] = True

        # Merge data
        dataset = Data(
            X=X,
            y=y,
            adjacency=adjacency,
            test_mask=test_mask,
            train_mask=train_mask,
            valid_mask=valid_mask
        )

        return dataset

    @staticmethod
    def read_data(file):
        """Read raw data using different methods

            Input:
            ------
            file:, string, file path to read

            Output:
            -------
            content: numpy array, file content

        """

        file_name = os.path.basename(file)
        if 'test.index' in file_name:
            content = np.genfromtxt(file, dtype='int64')
        else:
            content = pickle.load(open(file, 'rb'), encoding='latin1')
            if hasattr(content, 'toarray'):
                content = content.toarray()

        return content

    @staticmethod
    def build_adjacency(graph):
        """Build adjacency matrix based on graph structure

            Input:
            ------
            graph: dict, dictionary of adjacent nodes for each node

            Output:
            -------
            adjacency: numpy array, adjacency matrix

        """

        # Node index list for each edge, each pair of nodes represents an edge
        edge_index = []
        for src, dst in graph.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)

        # Remove duplicate edges
        sorted_edge_index = sorted(edge_index)
        edge_index = list(k for k, _ in groupby(sorted_edge_index))

        # Build adjacency matrix
        num_nodes = len(graph)
        num_edges = len(edge_index)
        edge_index = np.asarray(edge_index)
        adjacency = coo_matrix((
            np.ones(num_edges),
            (edge_index[:, 0], edge_index[:, 1])
        ), shape=(num_nodes, num_nodes), dtype=float)

        return adjacency

    @staticmethod
    def build_adjacency1(data):

        """
        Generate adjacency matrix.
        Returns:
            torch.Tensor: Adjacency matrix with shape (num_nodes, num_nodes), where num_nodes is the number of nodes.
                          Value is 1 if there is a connection between corresponding nodes, otherwise 0.
        """
        # Get number of nodes
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Initialize adjacency matrix
        adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int32)

        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        adjacency_matrix[src_nodes, dst_nodes] = 1
        adjacency_matrix[dst_nodes, src_nodes] = 1
        adjacency_matrix_coo = coo_matrix(adjacency_matrix.numpy())

        return adjacency_matrix_coo

    def __citeseer(self, tx, test_index, graph, num_nodes):
        """Special processing for citeseer dataset

            The citeseer test set is missing several nodes, which need to be removed from the adjacency matrix,
            and the test set node indices need to be updated

            Inputs:
            -------
            tx: numpy array, test set features
            test_index: numpy array, test set node index list
            graph: dict, graph network dictionary
            num_nodes: int, total number of nodes in the original graph

            Outputs:
            --------
            test_index: numpy array, updated test set node indices
            adjacency: sparse array, updated adjacency matrix

        """

        # Get missing node indices in the test set
        full_test = list(range(min(test_index), num_nodes))
        missing = [index for index in full_test if index not in test_index]

        # Reorder test set indices, reassign indices to test set nodes in order, and establish mapping between old and new indices
        new_test_index = list(range(min(test_index), min(test_index) + len(tx)))
        test_index_dict = {s: n for s, n in zip(sorted(test_index), new_test_index)}
        test_index = [test_index_dict[t] for t in test_index]

        # Rebuild graph after updating test set node indices
        new_graph = {}

        for key, values in graph.items():
            # Iterate over each center node and its neighbors in the original graph

            # Remove missing test set nodes from the neighbor list
            values = [v for v in values if v not in missing]
            if (key in missing) or (len(values) == 0):
                # If the center node is a missing test set node, or its neighbors only contain missing test set nodes,
                # then remove this center node and its neighbors
                continue

            if key in test_index_dict.keys():
                # Update test set node index
                key = test_index_dict[key]

            new_values = []
            for v in values:
                if v in test_index_dict.keys():
                    # Update test set node neighbor index
                    new_values.append(test_index_dict[v])
                else:
                    # Keep other node neighbor indices
                    new_values.append(v)

            # Add updated center node and its neighbors
            new_graph[key] = new_values

        # Build adjacency matrix using the updated graph
        adjacency = self.build_adjacency(new_graph)

        return test_index, adjacency
