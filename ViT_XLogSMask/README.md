# Vision Transformer with XLogSMask

This project implements image classification on the CIFAR-10 dataset using a Transformer with XLogSMask.

## Project Structure

```
ViT_XLogSMask/
├── configs/           # Configuration files
├── data/              # Dataset storage directory
├── models/            # Model definitions
├── utils/             # Utility functions
├── train.py           # Training script
├── evaluate.py        # Evaluation script
└── requirements.txt   # Dependency list
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python train.py --config configs/cifar10_config.yaml
```

## Evaluate Model

```bash
python evaluate.py --config configs/cifar10_config.yaml --model_path path/to/model.pth
```
