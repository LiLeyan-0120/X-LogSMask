import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os

class EarlyStopping:
    """Early stopping mechanism"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing learning rate scheduler with warm restarts"""
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1,
                 min_lr=0.001, warmup_steps=0, gamma=1.0, last_epoch=-1):

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * 
                    (1 + np.cos(np.pi * (self.step_in_cycle-self.warmup_steps) /
                                (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(np.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        return self.base_lrs

def mixup_data(x, y, alpha=1.0):
    """Mixup data augmentation

    Args:
        x: Input data
        y: Labels
        alpha: Beta distribution parameter

    Returns:
        mixed_x: Mixed input
        y_a, y_b: Original label pairs
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class Trainer:
    """Trainer class"""
    def __init__(self, model, optimizer, criterion, scheduler=None, device='cuda', 
                 use_amp=True, use_mixup=False, mixup_alpha=0.2, grad_clip=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.grad_clip = grad_clip
        self.scaler = GradScaler() if use_amp else None

    def train_epoch(self, train_loader):
        """Train one epoch

        Args:
            train_loader: Training data loader

        Returns:
            avg_loss: Average loss
            accuracy: Accuracy
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Mixup data augmentation
            if self.use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, self.mixup_alpha)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    if self.use_mixup:
                        loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                if self.use_mixup:
                    loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(outputs, targets)

            # Backward propagation
            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Statistics
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            if self.use_mixup:
                # For Mixup data, use mixed labels to calculate approximate accuracy
                # Calculate prediction accuracy for both labels based on mixing coefficient
                correct_a = (predicted == targets_a).float().sum().item()
                correct_b = (predicted == targets_b).float().sum().item()
                correct += lam * correct_a + (1 - lam) * correct_b
            else:
                correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/len(pbar):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def evaluate(self, test_loader):
        """Evaluate model

        Args:
            test_loader: Test data loader

        Returns:
            avg_loss: Average loss
            accuracy: Accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(test_loader, desc='Evaluating')

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{running_loss/len(pbar):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

        avg_loss = running_loss / len(test_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, test_loader, epochs, output_dir, patience=20, resume=None):
        """Train model

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of training epochs
            output_dir: Output directory
            patience: Early stopping patience
            resume: Path to resume training checkpoint

        Returns:
            best_acc: Best test accuracy
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Early stopping mechanism
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        # Resume training
        start_epoch = 0
        best_acc = 0.0
        if resume:
            if os.path.isfile(resume):
                print(f"Loading checkpoint '{resume}'")
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                best_acc = checkpoint['best_acc']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.scheduler is not None and 'scheduler' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                print(f"Loaded checkpoint '{resume}' (epoch {start_epoch})")

        # Training loop
        for epoch in range(start_epoch, epochs):
            # Train one epoch
            train_loss, train_acc = self.train_epoch(train_loader)

            # Evaluate
            test_loss, test_acc = self.evaluate(test_loader)

            # Print training information
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

            # Early stopping check
            if early_stopping(test_loss, self.model):
                print(f"Early stopping triggered, stopping training at epoch {epoch+1}")
                break

            # Save best model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            self._save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': best_acc,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            }, is_best, output_dir)

        print(f'Best test accuracy: {best_acc:.2f}%')
        return best_acc
    
    def _save_checkpoint(self, state, is_best, output_dir):
        """Save model checkpoint

        Args:
            state: Dictionary containing model state, optimizer state, etc.
            is_best: Whether this is the best model
            output_dir: Output directory
        """
        filename = os.path.join(output_dir, 'checkpoint.pth')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(output_dir, 'model_best.pth')
            torch.save(state, best_filename)
        return best_acc

    def _save_checkpoint(self, state, is_best, output_dir):
        """Save model checkpoint

        Args:
            state: Dictionary containing model state, optimizer state, etc.
            is_best: Whether this is the best model
            output_dir: Output directory
        """
        filename = os.path.join(output_dir, 'checkpoint.pth')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(output_dir, 'model_best.pth')
            torch.save(state, best_filename)
