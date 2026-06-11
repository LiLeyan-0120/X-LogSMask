import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import optuna
import torch
import gc
from script.dataset import LinkPredDataset
from script.utils import load_config
from script.pipeline import Pipeline
from functools import partial


def objective(trial, data_name):
    try:
        # Dataset root directory
        dataset_root = '../../Dataset'
        base_config = load_config(config_file='config.yaml')
        data_config = base_config[data_name].copy()

        # Hyperparameter suggestions
        trial_params = {
            "model": {
                "dropout": trial.suggest_float("dropout", 0.0, 0.6),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [48, 192, 384, 528]),
                "num_heads": trial.suggest_categorical("num_heads", [3, 4, 6, 8]),
                "dim_feedforward": trial.suggest_categorical("dim_feedforward", [1, 2, 3]),
                "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3])
            },
            "hyper": {
                "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
                "weight_decay": 0.005,
                "epochs": 3000,
                "patience": 100,
                "neg_per_pos": trial.suggest_categorical("neg_per_pos", [1, 3, 5, 8])
            },
            "sampling": {
                "train_times_per_epoch": trial.suggest_categorical("train_times_per_epoch", [16, 40, 60, 80])
            }
        }

        # Merge configuration: override base config with trial parameters
        for section, params in trial_params.items():
            if section in data_config:
                data_config[section].update(params)
            else:
                data_config[section] = params

        # Data acquisition and preprocessing
        dataset = LinkPredDataset(data_name, dataset_root, **data_config)

        pipeline = Pipeline(**data_config)
        pipeline.train(dataset)
        # Test set accuracy
        loss, test_result = pipeline.predict(dataset, 'test')

        return test_result[next(iter(test_result))]

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clean up GPU memory
        gc.collect()  # Force garbage collection


if __name__ == "__main__":

    # dataset_list = ['cora', 'citeseer', 'pubmed', 'Collab', 'PPA', 'Citation2', 'DDI']
    dataset_list = ['cora', 'citeseer', 'pubmed']

    for data_name in dataset_list:

        output_file = r'param_results_edge_' + data_name + '.txt'
        with open(output_file, 'w') as f:
            # Create Optuna study
            study = optuna.create_study(
                study_name=data_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()  # Automatically prune poorly performing trials
            )

            # Start optimization
            study.optimize(partial(objective, data_name=data_name), n_trials=50, show_progress_bar=False)

            # Output results
            print("=" * 50)
            print("Best hyperparameters:")
            for key, value in study.best_params.items():
                print(f"{key:>15}: {value}")

            print(f"\nBest validation accuracy: {study.best_value:.4f}")

            # Visualization results
            # fig1 = optuna.visualization.plot_param_importances(study)
            # fig1.show()
            # fig2 = optuna.visualization.plot_optimization_history(study)
            # fig2.show()

            f.write(f"Data: {data_name}\n")
            f.write("Best Parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Best Validation Accuracy: {study.best_value:.4f}\n")
            f.write("=" * 50 + "\n")
