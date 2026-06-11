import os
import yaml
import torch
import numpy as np

def load_config(config_path):
    """Load configuration file

    Args:
        config_path (str): Config file path

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

def set_seed(seed):
    """Set random seed

    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
