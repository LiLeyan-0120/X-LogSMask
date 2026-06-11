import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_config, set_seed, get_cifar10_loaders, Trainer, CosineAnnealingWarmupRestarts
from models import VisionTransformerWithXLogSMask, VisionTransformer

def get_optimizer(model, config):
    """Get optimizer"""
    optimizer_name = config.get('optimizer', 'adamw')  # Default to AdamW
    if optimizer_name == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.05)
        )
    elif optimizer_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=0.9,
            weight_decay=config.get('weight_decay', 0.0001)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, config):
    """Get learning rate scheduler"""
    scheduler_name = config.get('scheduler', 'cosine')  # Default to cosine annealing
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config['lr'] * 0.01
        )
    elif scheduler_name == 'cosine_warmup':
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=config['epochs'] // 2,
            max_lr=config['lr'],
            min_lr=config['lr'] * 0.01,
            warmup_steps=config['epochs'] // 10
        )
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['epochs'] // 3,
            gamma=0.1
        )
    else:
        return None

def get_criterion(config):
    """Get loss function"""
    if 'label_smoothing' in config and config['label_smoothing'] > 0:
        return nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    else:
        return nn.CrossEntropyLoss()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vision Transformer with X-LogSMask Training')
    # parser.add_argument('--config', type=str, default='configs/cifar10_config.yaml', help='Config file path')
    parser.add_argument('--config', type=str, default='configs/vit_config.yaml', help='Config file path')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume training checkpoint')

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config.get('seed', 42))

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Prepare data
    train_loader, test_loader, num_classes = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=config['batch_size'],
        img_size=config.get('img_size', 32)
    )

    # Create model based on configuration
    model_type = config.get('model_type', 'vit_xlogsmask')

    if model_type == 'vit':
        model = VisionTransformer(
            img_size=config.get('img_size', 32),
            patch_size=config.get('patch_size', 4),
            in_channels=3,
            num_classes=num_classes,
            embed_dim=config.get('embed_dim', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1)
        ).to(device)
    else:  # Default to vit_xlogsmask
        model = VisionTransformerWithXLogSMask(
            img_size=config.get('img_size', 32),
            patch_size=config.get('patch_size', 4),
            in_channels=3,
            num_classes=num_classes,
            embed_dim=config.get('embed_dim', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            use_xlogsmask=config.get('use_xlogsmask', True)
        ).to(device)

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Get optimizer, scheduler and loss function
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    criterion = get_criterion(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        use_amp=config.get('use_amp', True),
        use_mixup=config.get('use_mixup', False),
        mixup_alpha=config.get('mixup_alpha', 0.2),
        grad_clip=config.get('grad_clip', None)
    )

    # Train model
    best_acc = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config['epochs'],
        output_dir=args.output_dir,
        patience=config.get('patience', 20),
        resume=args.resume
    )

    print(f'Training complete, best test accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
