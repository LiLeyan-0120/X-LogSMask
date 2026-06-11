# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
from script.dataset import LinkPredDataset
from script.utils import load_config
from script.pipeline import Pipeline


def train_and_test(data, dataset_root, config):
    """Model training and testing

        Train and test the model using given data and configuration

        Inputs:
        -------
        data: string, Dataset name to use, ['DD', 'NCI1', 'PROTEINS']
        dataset_root: string, Root folder path for saving datasets
        config: dict, Parameter configuration

    """

    # Data acquisition and preprocessing
    dataset = LinkPredDataset(data, dataset_root, **config[data])

    # Train model
    pipeline = Pipeline(**config[data])
    pipeline.train(dataset)

    # Test set accuracy
    test_loss, test_result = pipeline.predict(dataset, 'test')

    results_str = ", ".join([f"{k}:{v:.4f}" for k, v in test_result.items()])
    print('[{}]-[TestLoss:{:.4f}]-[ValidMetrics:{}]\n'.format(
        data.upper(), test_loss, results_str))

    return


if __name__ == '__main__':

    # Dataset root directory
    dataset_root = '../../Dataset'

    # Load global configuration
    config = load_config(config_file='config.yaml')

    # Train and test the model using DD dataset
    # train_and_test('cora', dataset_root, config)
    # Cora Test Accuracy: 0.4

    # Train and test the model using DD dataset
    # train_and_test('citeseer', dataset_root, config)
    # Cora Test Accuracy: 0.824

    # Train and test the model using DD dataset
    # train_and_test('pubmed', dataset_root, config)
    # Cora Test Accuracy: 0.824

    # Train and test the model using DD dataset
    # train_and_test('collab', dataset_root, config)
    # Cora Test Accuracy: 0.824

    # Train and test the model using DD dataset
    # train_and_test('PPA', dataset_root, config)
    # Cora Test Accuracy: 0.824

    # Train and test the model using DD dataset
    # train_and_test('Citation2', dataset_root, config)
    # Cora Test Accuracy: 0.824

    # Train and test the model using DD dataset
    train_and_test('DDI', dataset_root, config)
    # Cora Test Accuracy: 0.824
