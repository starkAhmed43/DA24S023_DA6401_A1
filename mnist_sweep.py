import wandb
import argparse
import numpy as np
from tqdm.auto import tqdm
from dataloader import load_data
from train import hparam_search


np.random.seed(42)

# Configuration 1: Balanced, High-Accuracy Model
config1 = {
    "method": "grid",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "dataset": {"values": ["mnist"]},
        "epochs": {"values": [10]},
        "num_hidden_layers": {"values": [4]},
        "hidden_layer_size": {"values": [128]},
        "weight_decay": {"values": [0.0005]},
        "learning_rate": {"values": [0.001]},
        "loss": {"values": ["cross_entropy"]},
        "beta1": {"values": [0.9]},
        "beta2": {"values": [0.999]},
        "momentum": {"values": [0.9]},
        "epsilon": {"values": [1e-8]},
        "optimizer": {"values": ["adam"]},
        "batch_size": {"values": [32]},
        "weight_init": {"values": ["xavier"]},
        "activation": {"values": ["relu"]}
    }
}

# Configuration 2: Lightweight, Efficient Model
config2 = {
    "method": "grid",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "dataset": {"values": ["mnist"]},
        "epochs": {"values": [5]},
        "num_hidden_layers": {"values": [3]},
        "hidden_layer_size": {"values": [64]},
        "weight_decay": {"values": [0]},
        "learning_rate": {"values": [0.0001]},
        "loss": {"values": ["cross_entropy"]},
        "beta1": {"values": [0.99]},
        "beta2": {"values": [0.9999]},
        "momentum": {"values": [0.99]},
        "epsilon": {"values": [1e-7]},
        "optimizer": {"values": ["rmsprop"]},
        "batch_size": {"values": [16]},
        "weight_init": {"values": ["xavier"]},
        "activation": {"values": ["relu"]}
    }
}

# Configuration 3: Deep Regularized Model
config3 = {
    "method": "grid",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "dataset": {"values": ["mnist"]},
        "epochs": {"values": [10]},
        "num_hidden_layers": {"values": [5]},
        "hidden_layer_size": {"values": [128]},
        "weight_decay": {"values": [0.05]},
        "learning_rate": {"values": [0.001]},
        "loss": {"values": ["cross_entropy"]},
        "beta1": {"values": [0.9]},
        "beta2": {"values": [0.9999]},
        "momentum": {"values": [0.9]},
        "epsilon": {"values": [1e-8]},
        "optimizer": {"values": ["adam"]},
        "batch_size": {"values": [64]},
        "weight_init": {"values": ["xavier"]},
        "activation": {"values": ["relu"]}
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an NN on MNIST with 3 hyperparameter configs.")
    parser.add_argument("-wp", "--wandb_project", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    
    args = parser.parse_args()

    sweep_kwargs = {}
    if args.wandb_project:
        sweep_kwargs["project"] = args.wandb_project
    if args.wandb_entity:
        sweep_kwargs["entity"] = args.wandb_entity

    X_train, y_train, X_val, y_val, X_test, y_test = load_data("mnist")
    for config in tqdm([config1, config2, config3], desc="Sweeping configurations", unit="config"):
        
        sweep_id = wandb.sweep(config, **sweep_kwargs)
        wandb.agent(sweep_id, function=lambda: hparam_search(X_train, y_train, X_val, y_val), count=1)
        wandb.finish()