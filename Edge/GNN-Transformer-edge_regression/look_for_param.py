import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import optuna
import torch
import gc
from script.dataset import Dataset
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
                "hidden_dim": trial.suggest_categorical("hidden_dim", [48, 96, 192, 384]),
                "num_heads": trial.suggest_categorical("num_heads", [3, 4, 6, 8]),
                "dim_feedforward": trial.suggest_categorical("dim_feedforward", [1, 2, 3]),
                "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3])
            },
            "hyper": {
                "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                "epochs": 1000,
                "patience": 10
            },
            "sampling": {
                "train_times_per_epoch": trial.suggest_categorical("train_times_per_epoch", [5, 10, 20, 40])
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

        pipeline = Pipeline(**data_config)
        pipeline.train(dataset)
        # Test set evaluation
        loss, test_result = pipeline.predict(dataset, 'test')

        # Store test metrics in trial's user_attrs for later use
        trial.set_user_attr('test_MAE', test_result.get('test_MAE'))
        trial.set_user_attr('test_RMSE', test_result.get('test_RMSE'))

        return test_result.get('test_MAE')

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Force garbage collection


if __name__ == "__main__":

    # Edge regression dataset list
    # dataset_list = ['epic-games-plr', 'air-traffic-2019-rlr', 'air-traffic-2015-rlr']
    dataset_list = ['epic-games-plr']

    for data_name in dataset_list:

        output_file = r'param_results_edge_' + data_name + '.txt'
        with open(output_file, 'w') as f:
            # Create Optuna study
            study = optuna.create_study(
                study_name=data_name,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()  # Auto-prune poorly performing trials
            )

            # Start optimization
            study.optimize(partial(objective, data_name=data_name), n_trials=5, show_progress_bar=False)

            # Output results
            print("=" * 50)
            print("Best hyperparameters:")
            for key, value in study.best_params.items():
                print(f"{key:>15}: {value}")

            print(f"\nBest validation MAE: {study.best_value:.4f}")

            # Get test metrics from the best trial
            best_trial = study.best_trial
            test_mae = best_trial.user_attrs.get('test_MAE')
            test_rmse = best_trial.user_attrs.get('test_RMSE')

            # Visualization results
            # fig1 = optuna.visualization.plot_param_importances(study)
            # fig1.show()
            # fig2 = optuna.visualization.plot_optimization_history(study)
            # fig2.show()

            f.write(f"Data: {data_name}\n")
            f.write("Best Parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Test MAE: {test_mae:.4f}\n")
            f.write(f"Test RMSE: {test_rmse:.4f}\n")
            f.write("=" * 50 + "\n")
