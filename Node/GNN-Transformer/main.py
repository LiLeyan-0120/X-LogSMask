# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
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
    pipeline.train(prep_dataset)

    # Test set accuracy
    test_acc = pipeline.predict(prep_dataset, 'test')
    print('[{}] Test Accuracy: {:.3f}\n'.format(data.upper(), test_acc))

    return


if __name__ == '__main__':

    # Dataset root directory
    dataset_root = '../../Dataset'

    # Load global configuration
    config = load_config(config_file='config.yaml')

    # Train and test model using Cora dataset
    # train_and_test('cora', dataset_root, config)
    # Cora Test Accuracy: 0.827

    # Train and test model using Citeseer dataset
    # train_and_test('Citeseer', dataset_root, config)
    # Citeseer Test Accuracy: 0.717

    # Train and test model using Pubmed dataset
    # train_and_test('pubmed', dataset_root, config)
    # Pubmed Test Accuracy: 0.793

    # Train and test model using Pubmed dataset
    # train_and_test('Computers', dataset_root, config)
    # Pubmed Test Accuracy: 0.793

    # Train and test model using Citeseer dataset
    # train_and_test('Photo', dataset_root, config)
    # Citeseer Test Accuracy: 0.717

    # Train and test model using Pubmed dataset
    # train_and_test('CS', dataset_root, config)
    # Pubmed Test Accuracy: 0.793

    # Train and test model using Pubmed dataset
    train_and_test('Physics', dataset_root, config)
    # Pubmed Test Accuracy: 0.793

    # Train and test model using Pubmed dataset
    # train_and_test('WikiCS', dataset_root, config)
    # Pubmed Test Accuracy: 0.793

