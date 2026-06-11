# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import optuna
import torch
import gc
from functools import partial

from script.dataset import Dataset
from script.utils import load_config
from script.pipeline import Pipeline

def objective(trial, data_name):
    try:
        # Dataset root directory
        dataset_root = '../../Dataset'
        base_config = load_config(config_file='config.yaml')
        data_config = base_config[data_name].copy()  # Create a copy to avoid modifying original config

        # Hyperparameter suggestions
        trial_params = {
            "model": {
                "dropout": trial.suggest_float("dropout", 0, 0.6),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [48, 96, 192, 384]),
                "num_heads": trial.suggest_categorical("num_heads", [2, 3, 4, 6, 8]),
                "dim_feedforward": trial.suggest_categorical("dim_feedforward", [1, 2, 3, 4]),
                "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3])
            },
            "hyper": {
                "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "weight_decay": 0.0005,
                # Adjust training epochs to speed up tuning
                "epochs": 1000,
                "patience": 100,
                "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64])
            }
        }

        # Merge configuration: override base config with trial parameters
        for section, params in trial_params.items():
            if section in data_config:
                data_config[section].update(params)
            else:
                data_config[section] = params

        # Data acquisition and preprocessing
        dataset = Dataset(data_name, dataset_root, **data_config)

        pipeline = Pipeline(data_name, **data_config)
        if data_name in ['ZINC_full', 'ZINC']:
            pipeline.criterion = torch.nn.L1Loss(reduction='mean')
            pipeline.train2(dataset)
            # Test set accuracy
            loss = pipeline.predict2(dataset, 'test')
            return loss
        else:
            pipeline.train(dataset)
            # Test set accuracy
            loss, test_acc = pipeline.predict(dataset, 'test')

            return test_acc

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Force garbage collection


if __name__ == "__main__":

    # dataset_list = ['DD', 'NCI1', 'PROTEINS', 'MUTAG', 'ZINC_full', 'ZINC', 'COLLAB', 'IMDB-BINARY', 'MOLHIV']
    dataset_list = ['COLLAB', 'IMDB-BINARY', 'MUTAG', 'PROTEINS', 'DD', 'ZINC', 'NCI1']

    for data_name in dataset_list:

        output_file = r'param_results_graph_' + data_name + '.txt'

        with open(output_file, 'w') as f:
            # Create Optuna study
            if data_name in ['ZINC_full', 'ZINC']:
                study = optuna.create_study(
                    study_name=data_name,
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner()  # Auto-prune poorly performing trials
                )
            else:
                study = optuna.create_study(
                    study_name=data_name,
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner()  # Auto-prune poorly performing trials
                )

            # Start optimization (60 trials)
            study.optimize(partial(objective, data_name=data_name), n_trials=60, show_progress_bar=False)

            # Output results
            print("=" * 50)
            print("Best hyperparameters:")
            for key, value in study.best_params.items():
                print(f"{key:>15}: {value}")

            print(f"\nBest validation accuracy: {study.best_value:.4f}")

            f.write(f"Data: {data_name}\n")
            f.write("Best Parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Best Validation Accuracy: {study.best_value:.4f}\n")
            f.write("=" * 50 + "\n")
