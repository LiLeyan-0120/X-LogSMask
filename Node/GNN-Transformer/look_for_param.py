import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import optuna
import torch
import gc
from script.dataset import Dataset
from script.prepare import prepare
from script.utils import load_config
from script.pipeline import Pipeline
from functools import partial


def objective(trial, data_name):
    try:
        # Dataset root directory
        dataset_root = '../../Dataset'
        base_config = load_config(config_file='config.yaml')
        data_config = base_config[data_name].copy()  # Create a copy to avoid modifying the original configuration

        # Data acquisition and preprocessing
        dataset = Dataset(data_name, dataset_root)
        prep_dataset = prepare(dataset)

        # Hyperparameter suggestions
        trial_params = {
            "model": {
                "dropout": trial.suggest_float("dropout", 0.2, 0.8),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
                "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
                "dim_feedforward": trial.suggest_categorical("dim_feedforward", [1, 2, 3, 4]),
                "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3, 4])
            },
            "hyper": {
                "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "weight_decay": 0.0005,
                # Adjust training epochs to speed up tuning
                "epochs": 3000,
                "patience": 300
            },
            "sampling": {
                "num_nodes": trial.suggest_categorical("num_nodes", [800, 1600, 2400]),
                "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32])
            }
        }

        # Merge configuration: override base config with trial parameters
        for section, params in trial_params.items():
            if section in data_config:
                data_config[section].update(params)
            else:
                data_config[section] = params

        pipeline = Pipeline(**data_config)
        pipeline.train(prep_dataset)

        # Test set accuracy
        test_acc = pipeline.predict(prep_dataset, 'test')

        return test_acc

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Force garbage collection


if __name__ == "__main__":

    # dataset_list = ['cora', 'citeseer', 'pubmed', 'Photo', 'CS', 'Physics', 'WikiCS', 'Computers']
    # dataset_list = ['cora', 'citeseer', 'Photo']
    dataset_list = ['WikiCS', 'pubmed', 'CS', 'Physics', 'Computers']
    output_file = r'param_results_node_0901.txt'

    with open(output_file, 'w') as f:

        for data_name in dataset_list:
            # Create Optuna study
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()  # Automatically prune poorly performing trials
            )

            # Start optimization
            study.optimize(partial(objective, data_name=data_name), n_trials=80, show_progress_bar=False)

            # Output results
            print("=" * 50)
            print("Best hyperparameters:")
            for key, value in study.best_params.items():
                print(f"{key:>15}: {value}")

            print(f"\nBest validation accuracy: {study.best_value:.4f}")

            # Visualization results
            # fig = optuna.visualization.plot_param_importances(study)
            # fig.show()
            # fig2 = optuna.visualization.plot_optimization_history(study)
            # fig2.show()

            f.write(f"Data: {data_name}\n")
            f.write("Best Parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Best Validation Accuracy: {study.best_value:.4f}\n")
            f.write("=" * 50 + "\n")
