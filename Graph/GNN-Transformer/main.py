# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
from script.dataset import Dataset
from script.utils import load_config
from script.pipeline import Pipeline
import torch


def train_and_test(data, dataset_root, config):
    """Model training and testing

        Train and test the model using given data and configuration

        Inputs:
        -------
        data: string, name of the dataset to use, ['DD', 'NCI1', 'PROTEINS']
        dataset_root: string, root folder path for saving datasets
        config: dict, parameter configuration

    """

    # Data acquisition and preprocessing
    dataset = Dataset(data, dataset_root, **config[data])

    # Train model
    pipeline = Pipeline(data, **config[data])

    if data in ['ZINC_full', 'ZINC']:
        pipeline.criterion = torch.nn.L1Loss(reduction='mean')
        pipeline.train2(dataset)
        # Test set accuracy
        test_loss = pipeline.predict2(dataset, 'test')
        print('[{}]-[TestLoss:{:.4f}]\n'.format(
            data.upper(), test_loss))
    else:
        pipeline.train(dataset)
        # Test set accuracy
        test_loss, test_acc = pipeline.predict(dataset, 'test')
        print('[{}]-[TestLoss:{:.4f}]-[TestAcc:{:.3f}]\n'.format(
            data.upper(), test_loss, test_acc))

    return


if __name__ == '__main__':

    # Dataset root directory
    dataset_root = '../../Dataset'

    # Load global configuration
    config = load_config(config_file='config.yaml')

    # Train and test model using DD dataset
    # train_and_test('DD', dataset_root, config)
    # DD Test Accuracy: 0.824

    # Train and test model using NCI1 dataset
    train_and_test('NCI1', dataset_root, config)
    # PROTEINS Test Accuracy: 0.718

    # Train and test model using PROTEINS dataset
    # train_and_test('PROTEINS', dataset_root, config)
    # PROTEINS Test Accuracy: 0.807

    # Train and test model using MOLHIV dataset
    # train_and_test('MOLHIV', dataset_root, config)
    # PROTEINS Test Accuracy: 0.807

    # Train and test model using MUTAG dataset
    # train_and_test('MUTAG', dataset_root, config)
    # MUTAG Test Accuracy: 0.865

    # Train and test model using COLLAB dataset
    # train_and_test('COLLAB', dataset_root, config)
    # COLLAB Test Accuracy: 0.820

    # Train and test model using ZINC dataset
    # train_and_test('ZINC', dataset_root, config)
    # COLLAB Test Accuracy: 0.820
