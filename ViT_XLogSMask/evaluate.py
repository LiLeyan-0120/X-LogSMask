import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import load_config, get_cifar10_loaders
from models import VisionTransformerWithXLogSMask, VisionTransformer

def evaluate(model, test_loader, criterion, device, use_amp=True):
    """Evaluate model

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device for evaluation
        use_amp: Whether to use automatic mixed precision

    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Use tqdm to show progress bar
    pbar = tqdm(test_loader, desc='Evaluating')

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/len(test_loader):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vision Transformer with X-LogSMask Evaluation')
    parser.add_argument('--config', type=str, default='configs/cifar10_config.yaml', help='Config file path')
    parser.add_argument('--model_path', type=str, required=True, help='Model weights path')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory')

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Prepare data
    _, test_loader, num_classes = get_cifar10_loaders(
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

    # Load model weights
    if os.path.isfile(args.model_path):
        print(f"Loading model weights '{args.model_path}'")
        checkpoint = torch.load(args.model_path, map_location=device)

        # Handle different weight file formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print("Model weights loaded successfully")
    else:
        raise FileNotFoundError(f"Model weights file '{args.model_path}' does not exist")

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate model
    test_loss, test_acc = evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=config.get('use_amp', True)
    )

    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    main()
