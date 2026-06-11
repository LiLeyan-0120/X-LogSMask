
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .model import Transformer
from .utils import compute_normal_adjacency


class Pipeline(object):

    def __init__(self, data_name, **params):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_name = data_name
        self.__init_environment(params['random_state'])
        self.__build_model(**params['model'])
        self.__build_components(**params['hyper'])

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
            model_params: dict, model related parameters

        """

        # Load model
        self.model = Transformer(**model_params)
        if self.data_name == 'ZINC':
            self.model.src_embedding = nn.Embedding(model_params['input_dim'], model_params['hidden_dim'])
        self.model.to(self.device)

        return

    def __build_components(self, **hyper_params):
        """Load components

            Input:
            ------
            hyper_params: dict, hyperparameters

        """

        self.epochs = hyper_params['epochs']
        self.patience = hyper_params['patience']
        self.batch_size = hyper_params['batch_size']

        # Learning rate warmup parameters
        self.warmup_epochs = hyper_params.get('warmup_epochs', 100)
        self.lr = hyper_params['lr']

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define optimizer
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=hyper_params['weight_decay']
        )

        # Create learning rate warmup scheduler
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / self.warmup_epochs)
        )

        return

    def train(self, dataset):
        """Train model

            Input:
            ------
            dataset: Dataset, contains test, valid, and train

        """
        # Training set batch loader
        train_loader = DataLoader(
            dataset=dataset.train,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Track best model on validation set
        best_model = None

        # Track best validation accuracy
        best_valid_metric = 0

        # Count epochs after best validation result
        epochs_after_best = 0

        for epoch in range(self.epochs):
            # Model training mode
            self.model.train()

            # Track loss of all batches in each epoch
            epoch_losses, y_true, y_pred, y_pos_probs = [], [], [], []

            for i, data in enumerate(train_loader):
                # Model output
                data = data.to(self.device)
                X, graph, batch = data.x, data.edge_index, data.batch
                adjacency = to_dense_adj(graph, batch)
                x0, batch = to_dense_batch(X, batch)
                adjacency = compute_normal_adjacency(adjacency)

                logits = self.model(x0.float(), adjacency)

                if self.data_name == 'MOLHIV':
                    data.y = data.y.squeeze(-1)

                # Calculate loss function
                loss = self.criterion(logits, data.y)
                epoch_losses.append(loss.item())
                predict_y = logits.max(1)[1]
                y_pred.extend(predict_y.cpu().numpy())
                true_y = data.y.cpu().numpy().flatten()
                y_true.extend(true_y)
                probs = torch.softmax(logits, dim=1)
                pos_probs = probs[:, 1].detach().cpu().numpy()
                y_pos_probs.extend(pos_probs)

                # Backward propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Calculate mean loss across all batches in the epoch
            epoch_loss = np.mean(epoch_losses)

            if self.data_name == 'MOLHIV':
                # Calculate AUROC (handle single-class case)
                if len(np.unique(y_true)) < 2:
                    train_metric = 0.5
                else:
                    train_metric = roc_auc_score(y_true, y_pos_probs)
                metric_name = 'AUROC'
            else:
                # Calculate multi-class accuracy
                train_metric = accuracy_score(y_true, y_pred)
                metric_name = 'Acc'

            # Update learning rate warmup scheduler
            self.scheduler.step()

            valid_loss, valid_metric = self.predict(dataset, 'valid')

            # Print log
            print(
                f'[Epoch:{epoch:03d}]-[TrainLoss:{epoch_loss:.4f}]-[Train{metric_name}:{train_metric:.4f}]-[Valid{metric_name}:{valid_metric:.4f}]-[LR:{self.optimizer.param_groups[0]["lr"]:.6f}]')

            if valid_metric > best_valid_metric:
                best_model = copy.deepcopy(self.model)
                best_valid_metric = valid_metric
                epochs_after_best = 0
            else:
                epochs_after_best += 1

            if epochs_after_best == self.patience:
                # Early stopping condition met
                self.model = best_model
                # torch.save(self.model, "result/model_PROTEINS.pth")
                break

        return

    @torch.no_grad()
    def predict(self, dataset, split='train'):
        """Model prediction

            Inputs:
            -------
            dataset: Dataset, contains test, valid, and train
            split: string, dataset split to use

            Output:
            -------
            loss: float, average loss across dataset samples
            accuracy: float, node classification accuracy

        """

        # Model inference mode
        self.model.eval()

        # Dataset split to use
        if split == 'train':
            eval_dataset = dataset.train
        elif split == 'valid':
            eval_dataset = dataset.valid
        else:  # split == 'test'
            eval_dataset = dataset.test

        # Dataset sample loader, 1 sample per batch
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=1, shuffle=False
        )

        # Track loss, labels, and predicted classes for each sample
        losses, y_true, y_pred, y_pos_probs = [], [], [], []
        for i, data in enumerate(eval_loader):
            # Model output
            data = data.to(self.device)
            X, graph, batch = data.x, data.edge_index, data.batch
            adjacency = to_dense_adj(graph, batch)
            x0, batch = to_dense_batch(X, batch)
            adjacency = compute_normal_adjacency(adjacency)

            logits = self.model(x0.float(), adjacency)

            if self.data_name == 'MOLHIV':
                data.y = data.y.squeeze(-1)

            loss = self.criterion(logits, data.y)
            losses.append(loss.item())

            true_y = data.y.cpu().numpy().flatten()  # Convert to 1D array (e.g., [batch_size])
            y_true.extend(true_y)
            predict_y = logits.max(1)[1]
            y_pred.extend(predict_y.cpu().numpy())
            probs = torch.softmax(logits, dim=1)
            pos_probs = probs[:, 1].detach().cpu().numpy()
            y_pos_probs.extend(pos_probs)

        # Calculate average loss across all samples
        loss = np.mean(losses)

        # Calculate accuracy
        if self.data_name == 'MOLHIV':
            if len(np.unique(y_true)) < 2:
                metric = 0.5
            else:
                metric = roc_auc_score(y_true, y_pos_probs)
        else:
            metric = accuracy_score(y_true, y_pred)
        return loss, metric

    def train2(self, dataset):
        """Train model

            Input:
            ------
            dataset: Dataset, contains test, valid, and train

        """
        # Training set batch loader
        train_loader = DataLoader(
            dataset=dataset.train,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Track best model on validation set
        best_model = None

        # Track best validation loss
        best_valid_loss = 100

        # Count epochs after best validation result
        epochs_after_best = 0

        for epoch in range(self.epochs):
            # Model training mode
            self.model.train()

            epoch_losses = []

            for i, data in enumerate(train_loader):

                data = data.to(self.device)
                X, graph, batch = data.x, data.edge_index, data.batch
                adjacency = to_dense_adj(graph, batch)
                x0, batch = to_dense_batch(X, batch)
                adjacency = compute_normal_adjacency(adjacency)

                logits = self.model(x0.squeeze(-1), adjacency)

                loss = self.criterion(logits.squeeze(-1), data.y)
                epoch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Calculate mean loss across all batches in the epoch
            # epoch_loss = np.mean(epoch_losses)

            # Update learning rate warmup scheduler
            self.scheduler.step()

            # Calculate validation loss
            valid_loss = self.predict2(dataset, 'valid')

            # print('[Epoch:{:03d}]-[TrainLoss:{:.4f}]-[ValidLoss:{:.4f}]'.format(
            #     epoch, epoch_loss, valid_loss))

            if valid_loss < best_valid_loss:
                best_model = copy.deepcopy(self.model)
                # Achieved best validation accuracy
                best_valid_loss = valid_loss
                # Reset epoch counter
                epochs_after_best = 0
            else:
                # Did not achieve best validation accuracy
                # Increment epoch counter
                epochs_after_best += 1

            if epochs_after_best == self.patience:
                # Early stopping condition met
                self.model = best_model
                break

        return

    @torch.no_grad()
    def predict2(self, dataset, split='train'):
        """Model prediction

            Inputs:
            -------
            dataset: Dataset, contains test, valid, and train
            split: string, dataset split to use

            Output:
            -------
            loss: float, average loss across dataset samples
            accuracy: float, node classification accuracy

        """

        # Model inference mode
        self.model.eval()

        # Dataset split to use
        if split == 'train':
            eval_dataset = dataset.train
        elif split == 'valid':
            eval_dataset = dataset.valid
        else:  # split == 'test'
            eval_dataset = dataset.test

        # Dataset sample loader, 1 sample per batch
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=1, shuffle=False
        )

        # Track loss of each sample
        losses = []
        for i, data in enumerate(eval_loader):
            # Model output
            data = data.to(self.device)
            X, graph, batch = data.x, data.edge_index, data.batch
            adjacency = to_dense_adj(graph, batch)
            x0, batch = to_dense_batch(X, batch)
            adjacency = compute_normal_adjacency(adjacency)

            logits = self.model(x0.squeeze(-1), adjacency)

            loss = self.criterion(logits.squeeze(-1), data.y)
            losses.append(loss.item())

        # Calculate average loss across all samples
        loss = np.mean(losses)

        return loss