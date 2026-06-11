
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
from script.dataset import Dataset
from script.utils import load_config
from script.pipeline import Pipeline


def train_and_test(data, dataset_root, config):
    """Model training and testing

        Train and test the model using given data and configuration

        Inputs:
        -------
        data: string, dataset name to use, ['epic-games-plr', 'air-traffic-2019-rlr', 'air-traffic-2015-rlr']
        dataset_root: string, root folder path for saving datasets
        config: dict, parameter configuration

    """

    # Data acquisition and preprocessing
    dataset = Dataset(data, dataset_root, **config[data])

    # Train model
    pipeline = Pipeline(**config[data])
    pipeline.train(dataset)

    # Test set evaluation
    test_loss, test_result = pipeline.predict(dataset, 'test')

    results_str = ", ".join([f"{k}:{v:.4f}" for k, v in test_result.items()])
    print('[{}]-[TestLoss:{:.4f}]-[TestMetrics:{}]'.format(
        data.upper(), test_loss, results_str))

    return


if __name__ == '__main__':

    # Dataset root directory
    dataset_root = '../../Dataset'

    # Load global configuration
    config = load_config(config_file='config.yaml')

    # Train and test model using epic-games-plr dataset
    train_and_test('epic-games-plr', dataset_root, config)

    # Train and test model using air-traffic-2019-rlr dataset
    # train_and_test('air-traffic-2019-rlr', dataset_root, config)

    # Train and test model using air-traffic-2015-rlr dataset
    # train_and_test('air-traffic-2015-rlr', dataset_root, config)
