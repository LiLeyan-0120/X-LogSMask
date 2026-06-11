from optuna.terminator.improvement.emmr import torch

from script.dataset import Dataset
from script.prepare import prepare
from script.utils import load_config
from script.pipeline import Pipeline


def train_and_test(data, dataset_root, config):
    """Model training and testing

        Train and test the model using given data and configuration

        Inputs:
        -------
        data: string, dataset name, ['cora', 'pubmed', 'citeseer']
        dataset_root: string, root folder path for saving datasets
        config: dict, parameter configuration

    """

    # Data acquisition and preprocessing
    dataset = Dataset(data, dataset_root)
    prep_dataset = prepare(dataset)

    # Train model
    pipeline = Pipeline(**config[data])
    pipeline.model = torch.load("result/model_Photo.pth")

    test_acc = pipeline.predict(prep_dataset, 'test')
    print('[{}] Test Accuracy: {:.3f}\n'.format(data.upper(), test_acc))

    return


if __name__ == '__main__':

    """
        Enable the plotting comments in layers when drawing
    """
    # Dataset root directory
    dataset_root = '../../Dataset'

    # Load global configuration
    config = load_config(config_file='config.yaml')

    # Train and test model using Cora dataset
    train_and_test('Photo', dataset_root, config)

