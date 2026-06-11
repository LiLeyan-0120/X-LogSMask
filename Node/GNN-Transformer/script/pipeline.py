
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from optuna.terminator.improvement.emmr import torch

from .model import TransformerEncoderModel, Transformer
from .utils import *
from torch.cuda.amp import autocast


class Pipeline(object):
    """GAT model training and prediction
    """

    def __init__(self, **params):
        """GNN-Transformer model training and prediction

            Load GNN-Transformer model, generate necessary training component instances

            Input:
            ------
            params: dict, model parameters and hyperparameters, format:
                    {
                        'sparse': False,
                        'random_state' 42,
                        'model': {
                            'input_dim': 1433,
                            'hidden_dim': 8,
                            'output_dim': 7,
                            'num_heads': 8,
                            'dropout': 0.6,
                            'alpha': 0.2
                        },
                        'hyper': {
                            'lr': 3e-3,
                            'epochs': 10,
                            'patience': 100,
                            'weight_decay': 5e-4
                        }
                    }

        """

        self.sparse = params['sparse']
        self.__init_environment(params['random_state'])
        self.__build_model(**params['model'])
        self.__build_components(**params['hyper'])
        self.sampling_config = params['sampling']

        return

    def __init_environment(self, random_state):
        """Initialize environment

            Input:
            ------
            random_state: int, random seed

        """

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return

    def __build_model(self, **model_params):
        """Load model

            Input:
            ------
            model_params: dict, model-related parameters

        """

        self.model = Transformer(**model_params)
        if torch.cuda.is_available():
            self.model.cuda()
        return

    def __build_components(self, **hyper_params):
        """Load components

            Input:
            ------
            hyper_params: dict, hyperparameters

        """

        self.epochs = hyper_params['epochs']
        self.patience = hyper_params['patience']

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define optimizer
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['weight_decay']
        )

        return

    def train(self, dataset):
        """Train model

            Input:
            ------
            dataset: Data, contains X, y, adjacency, test_index, train_index and valid_index

        """

        best_model = None

        # Track best validation accuracy
        best_valid_acc = 0

        # Count epochs after best validation result
        epochs_after_best = 0

        for epoch in range(self.epochs):
            # Model training mode
            self.model.train()

            batch_X, batch_adjacency, batch_labels, train_ind = \
                graph_sampling(dataset.X, dataset.adjacency, dataset.y, dataset.train_index, num_nodes=self.sampling_config['num_nodes'], batch_size=self.sampling_config['batch_size'])

            # batch_X=dataset.X.unsqueeze(0)
            # batch_adjacency=dataset.adjacency.unsqueeze(0)
            # batch_labels=dataset.y.unsqueeze(0)
            # train_ind=dataset.train_index
            logits = self.model(batch_X, batch_adjacency)

            onehot_labels = F.one_hot(batch_labels.view(-1), num_classes=logits.shape[-1]).view(logits.shape)
            loss = self.criterion(logits*(train_ind.unsqueeze(-1)), onehot_labels.float()*(train_ind.unsqueeze(-1)))
            # loss = self.criterion(logits[train_ind, :], onehot_labels.float()[train_ind, :])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.predict(dataset)
            # Calculate training set accuracy
            train_acc = self.train_acc
            # Calculate validation set accuracy
            valid_acc = self.valid_acc

            print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]-[ValidAcc:{:.4f}]'.format(
                epoch, loss, train_acc, valid_acc))

            if valid_acc >= best_valid_acc:
                best_model = copy.deepcopy(self.model)
                # Achieve best validation accuracy
                best_valid_acc = valid_acc
                # Reset epoch counter
                epochs_after_best = 0
            else:
                # Did not achieve best validation accuracy
                # Increment epoch counter
                epochs_after_best += 1

            if epochs_after_best == self.patience:
                # Early stopping condition met
                self.model = best_model
                # torch.save(self.model, "result/model_Photo.pth")
                break

        return

    def predict(self, dataset, split='train'):
        """Model prediction

            Inputs:
            -------
            dataset: Data, Data, contains X, y, adjacency, test_index,
                     train_index and valid_index
            split: string, nodes to predict

            Output:
            -------
            accuracy: float, node classification accuracy

        """

        # Model inference mode
        self.model.eval().to('cuda')

        # Node index
        if split == 'train':
            index = dataset.train_index
        elif split == 'valid':
            index = dataset.valid_index
        else:  # split == 'test'
            index = dataset.test_index

        # Get output for nodes to predict
        # logits = self.model(dataset.X, dataset.edges)
        with autocast():
            logits = self.model(dataset.X.unsqueeze(0), dataset.adjacency.unsqueeze(0))
        predict_y = logits[index].max(1)[1]

        # Calculate prediction accuracy
        y = dataset.y[index]
        accuracy = torch.eq(predict_y, y).float().mean()

        predict_y = logits[dataset.train_index].max(1)[1]
        y = dataset.y[dataset.train_index]
        self.train_acc = torch.eq(predict_y, y).float().mean()

        predict_y = logits[dataset.valid_index].max(1)[1]
        y = dataset.y[dataset.valid_index]
        self.valid_acc = torch.eq(predict_y, y).float().mean()

        return accuracy
