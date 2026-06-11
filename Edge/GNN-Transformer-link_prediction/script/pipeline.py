import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import to_dense_adj
from torch.utils.data import DataLoader

from .model import Transformer
from .utils import hit_ratio_at_k, mean_reciprocal_rank
from .dataset import GlobalUniform

class Pipeline(object):

    def __init__(self, **params):
        """Input:
            ------
            params: dict, Model parameters and hyperparameters

        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__init_environment(params['random_state'])
        self.__build_model(**params['model'])
        self.__build_components(**params['hyper'])
        self.metrics = params['metrics']
        self.sampling_config = params['sampling']

    def __init_environment(self, random_state):
        """Initialize environment

            Input:
            ------
            random_state: int, Random seed

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
            model_params: dict, Model-related parameters

        """

        # Load model
        self.model = Transformer(**model_params)
        self.model.to(self.device)

    def __build_components(self, **hyper_params):
        """Load components

            Input:
            ------
            hyper_params: dict, Hyperparameters

        """

        self.epochs = hyper_params['epochs']
        self.patience = hyper_params['patience']

        # Define loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Define optimizer
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['weight_decay']
        )

        self.neg_sampler = GlobalUniform(num_neg=hyper_params['neg_per_pos'])

        return

    def train(self, dataset):
        """Train model
        """

        # Track best validation model
        best_model = None
        # Model training mode
        self.model.train()

        best_valid_value = -1
        epochs_after_best = 0

        data = dataset.dataset.to(self.device)
        train_pos_edge = data.train_pos_edge_index
        total_nodes = data.num_nodes
        pos_adjacency = to_dense_adj(data.train_pos_edge_index, max_num_nodes=total_nodes)

        # Check if there is a learnable embedding layer in the data, if so add it to the optimizer
        if hasattr(data, 'embedding_layer') and data.embedding_layer is not None:
            # Create parameter list containing model parameters and embedding layer parameters
            all_params = list(self.model.parameters()) + list(data.embedding_layer.parameters())
            # Recreate optimizer including all parameters
            self.optimizer = optim.AdamW(
                params=all_params,
                lr=self.optimizer.param_groups[0]['lr'],
                weight_decay=self.optimizer.param_groups[0]['weight_decay']
            )
        train_times_per_epoch = self.sampling_config['train_times_per_epoch']

        for epoch in range(self.epochs):

            edge_dataloader = DataLoader(range(train_pos_edge.size(1)), train_pos_edge.size(1) // train_times_per_epoch, shuffle=True, drop_last=True)
            loss_list = []
            for edge_index in edge_dataloader:
                z = self.model(data.x.unsqueeze(0), pos_adjacency)

                epoch_train_pos_edge = train_pos_edge[:, edge_index]
                sources = epoch_train_pos_edge[0, :]
                batch_neg_v = self.neg_sampler(
                    num_nodes=total_nodes,
                    pos_edges=train_pos_edge,
                    sources=sources
                ).to(self.device)
                batch_neg_u = sources.unsqueeze(1).repeat(1, self.neg_sampler.num_neg).flatten()
                batch_neg_v_flat = batch_neg_v.flatten()
                epoch_train_neg_edge = torch.stack([batch_neg_u, batch_neg_v_flat], dim=1).t().to(self.device)

                pos_scores = self.model.predict_links(z, epoch_train_pos_edge)
                neg_scores = self.model.predict_links(z, epoch_train_neg_edge)

                # Calculate loss
                loss = self.criterion(
                    torch.cat([pos_scores, neg_scores]),
                    torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
                )
                loss_list.append(loss.item())

                # Backward propagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            valid_loss, valid_results = self.predict(data, split='valid')

            valid_results_str = ", ".join([f"{k}:{v:.4f}" for k, v in valid_results.items()])
            print('[Epoch:{:03d}]-[TrainLoss:{:.4f}]-[ValidLoss:{:.4f}]-[{}]'.format(
                epoch, np.mean(loss_list), valid_loss, valid_results_str))

            # Use the first metric (e.g., HR@K or MRR) as early stopping criterion, higher is better
            if len(valid_results) > 0:
                first_metric_name = list(valid_results.keys())[0]
                first_metric_value = valid_results[first_metric_name]
            else:
                first_metric_value = 0

            if first_metric_value > best_valid_value:
                best_model = copy.deepcopy(self.model)
                best_valid_value = first_metric_value
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
            dataset: Dataset, Contains test, valid and train
            split: string, Dataset split to use
        """

        # Model inference mode
        self.model.eval()


        # Used to track loss, labels and predicted classes for each sample
        results = {}

        if split == 'valid':
            pos_edge_index = data.valid_pos_edge_index
            neg_edge_index = data.valid_neg_edge_index
        elif split == 'test':
            data = data.dataset.to(self.device)
            pos_edge_index = data.test_pos_edge_index
            neg_edge_index = data.test_neg_edge_index

        train_pos_adj = to_dense_adj(data.train_pos_edge_index, max_num_nodes=data.num_nodes).to(self.device)
        z = self.model(data.x.unsqueeze(0), train_pos_adj)
        pos_scores = self.model.predict_links(z, pos_edge_index)
        neg_scores = self.model.predict_links(z, neg_edge_index)

        for metric in self.metrics:
            if metric.startswith('HR@'):
                k = int(metric.split('@')[1])
                hr = hit_ratio_at_k(pos_scores, neg_scores, k)
                results[f'{split}_{metric}'] = hr
            elif metric == 'MRR':
                mrr = mean_reciprocal_rank(pos_scores, neg_scores)
                results[f'{split}_MRR'] = mrr

        # Calculate validation/test loss
        all_scores = torch.cat([pos_scores, neg_scores], dim=0)
        all_labels = torch.cat([
            torch.ones_like(pos_scores),  # Positive edge label 1
            torch.zeros_like(neg_scores)  # Negative edge label 0
        ], dim=0)
        loss = self.criterion(all_scores, all_labels).item()

        return loss, results
