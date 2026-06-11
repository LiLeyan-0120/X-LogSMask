
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import to_dense_adj
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .model import Transformer

class Pipeline(object):

    def __init__(self, **params):
        """Input:
            ------
            params: dict, model parameters and hyperparameters

        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__init_environment(params['random_state'])
        self.__build_model(**params['model'])
        self.__build_components(**params['hyper'])
        self.sampling_config = params['sampling']

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

    def __build_model(self, **model_params):
        """Load model

            Input:
            ------
            model_params: dict, model-related parameters

        """

        # Load model
        self.model = Transformer(**model_params)
        self.model.to(self.device)

    def __build_components(self, **hyper_params):
        """Load components

            Input:
            ------
            hyper_params: dict, hyperparameters

        """

        self.epochs = hyper_params['epochs']
        self.patience = hyper_params['patience']

        # Define loss function - MSE loss for regression task
        self.criterion = nn.MSELoss()

        # Define optimizer
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['weight_decay']
        )

        return

    def train(self, dataset):
        """Train model
        """

        # Track best model on validation set
        best_model = None
        # Model training mode
        self.model.train()

        best_valid_loss = float('inf')
        epochs_after_best = 0

        data = dataset.dataset.to(self.device)
        train_edge_index = data.train_edge_index
        train_edge_weights = data.train_edge_attr
        total_nodes = data.num_nodes
        pos_adjacency = to_dense_adj(train_edge_index, max_num_nodes=total_nodes)

        # Check if data has learnable embedding layer, if so add it to optimizer
        if hasattr(data, 'embedding_layer') and data.embedding_layer is not None:
            # Create parameter list containing model and embedding layer parameters
            all_params = list(self.model.parameters()) + list(data.embedding_layer.parameters())
            # Recreate optimizer with all parameters
            self.optimizer = optim.AdamW(
                params=all_params,
                lr=self.optimizer.param_groups[0]['lr'],
                weight_decay=self.optimizer.param_groups[0]['weight_decay']
            )

        # Create training data loader
        edge_dataloader = DataLoader(range(train_edge_index.size(1)), batch_size=train_edge_index.size(1)//self.sampling_config['train_times_per_epoch'], shuffle=True)

        for epoch in range(self.epochs):
            loss_list = []
            for edge_idx in edge_dataloader:
                # Get current batch edges and weights
                batch_edge_index = train_edge_index[:, edge_idx]
                batch_edge_weights = train_edge_weights[edge_idx]

                # Forward propagation
                z = self.model(data.node_attr.unsqueeze(0), pos_adjacency)
                pred_weights = self.model.predict_edge_weights(z, batch_edge_index)

                # Compute loss
                loss = self.criterion(pred_weights, batch_edge_weights)
                loss_list.append(loss.item())

                # Backward propagation
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Validation set evaluation
            valid_loss, valid_results = self.predict(data, split='valid')

            valid_results_str = ", ".join([f"{k}:{v:.4f}" for k, v in valid_results.items()])
            print('[Epoch:{:03d}]-[TrainLoss:{:.4f}]-[ValidLoss:{:.4f}]-[{}]'.format(
                epoch, np.mean(loss_list), valid_loss, valid_results_str))

            # Early stopping check
            if valid_loss < best_valid_loss:
                best_model = copy.deepcopy(self.model)
                best_valid_loss = valid_loss
                epochs_after_best = 0
            else:
                epochs_after_best += 1

            if epochs_after_best == self.patience:
                # Early stopping condition met
                self.model = best_model
                break

    @torch.no_grad()
    def predict(self, data, split='valid'):
        """Model prediction

            Inputs:
            -------
            dataset: Dataset, contains test, valid, and train
            split: string, dataset split to use
        """

        # Model inference mode
        self.model.eval()

        # Track results
        results = {}

        if split == 'valid':
            edge_index = data.valid_edge_index
            edge_weights = data.valid_edge_attr
        elif split == 'test':
            data = data.dataset
            edge_index = data.test_edge_index
            edge_weights = data.test_edge_attr

        train_pos_adj = to_dense_adj(data.train_edge_index, max_num_nodes=data.num_nodes).to(self.device)
        z = self.model(data.node_attr.unsqueeze(0), train_pos_adj)
        pred_weights = self.model.predict_edge_weights(z, edge_index)

        # Compute regression metrics
        true_weights = edge_weights

        # Move predictions and ground truth to CPU for computation
        pred_weights_cpu = pred_weights.cpu().numpy()
        true_weights_cpu = true_weights.cpu().numpy()

        # Compute MSE and MAE
        mse = mean_squared_error(true_weights_cpu, pred_weights_cpu)
        mae = mean_absolute_error(true_weights_cpu, pred_weights_cpu)

        # Compute RMSE
        rmse = np.sqrt(mse)

        # Compute R-squared
        ss_res = np.sum((true_weights_cpu - pred_weights_cpu) ** 2)
        ss_tot = np.sum((true_weights_cpu - np.mean(true_weights_cpu)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        results[f'{split}_MSE'] = mse
        results[f'{split}_MAE'] = mae
        results[f'{split}_RMSE'] = rmse
        results[f'{split}_R2'] = r2

        # Compute loss
        loss = self.criterion(pred_weights, true_weights).item()

        return loss, results
