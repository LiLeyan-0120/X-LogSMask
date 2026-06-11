from optuna.terminator.improvement.emmr import torch

from script.dataset import Dataset
from script.utils import load_config
from script.pipeline import Pipeline


def train_and_test(data, dataset_root, config):
    """Model training and testing

        Train and test the model using given data and configuration

        Inputs:
        -------
        data: string, name of the dataset to use, ['cora', 'pubmed', 'citeseer']
        dataset_root: string, root folder path for saving datasets
        config: dict, parameter configuration

    """

    # Data acquisition and preprocessing
    dataset = Dataset(data, dataset_root, **config[data])

    # Train model
    pipeline = Pipeline(data, **config[data])
    pipeline.model = torch.load("result/model_PROTEINS.pth")

    test_loss, test_acc = pipeline.predict(dataset, 'test')
    print('[{}] Test Accuracy: {:.3f}\n'.format(data.upper(), test_acc))

    return


if __name__ == '__main__':

    """
        When plotting, uncomment the visualization code in layers.py
    """
    # Dataset root directory
    dataset_root = '../../Dataset'

    # Load global configuration
    config = load_config(config_file='config.yaml')

    # Train and test model using PROTEINS dataset
    train_and_test('PROTEINS', dataset_root, config)

