import wandb
import argparse
import numpy as np
import optimizer as opt
from tqdm.auto import tqdm
from keras.datasets import fashion_mnist
from neural_network import NeuralNetwork
from optimizer import OptimizerFactory

sweep_config = {
    "method": "bayes",  # Bayesian Optimization
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [50, 100]},
        "num_hidden_layers": {"values": [3, 4, 5]},
        "hidden_layer_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "beta1": {"values": [0.9, 0.99]},
        "beta2": {"values": [0.999, 0.9999]},
        "momentum": {"values": [0.9, 0.99]},
        "epsilon": {"values": [1e-8, 1e-7]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]}
    }
}

def cross_entropy_loss(y_hat_probs, y_true_classes):
    m = y_hat_probs.shape[0]  # Batch size
    y_true_one_hot = np.zeros_like(y_hat_probs)
    y_true_one_hot[np.arange(m), y_true_classes] = 1
    loss = -np.sum(y_true_one_hot * np.log(y_hat_probs + 1e-9)) / m  # Averaged over batch
    return loss

def train(model, X_train, y_train, X_val, y_val, optimizer, epochs=1000):    
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        train_probs = model.forward(X_train)
        train_loss = cross_entropy_loss(train_probs, y_train)
        train_accuracy = np.mean(np.argmax(train_probs, axis=1) == y_train)

        grad_W, grad_b = model.backward(X_train, train_probs, y_train)
        if isinstance(optimizer, opt.NAGOptimizer):
            optimizer.update(X_train, y_train)
        else:
            optimizer.update(grad_W, grad_b)

        val_probs = model.forward(X_val)
        val_loss = cross_entropy_loss(val_probs, y_val)
        val_accuracy = np.mean(np.argmax(val_probs, axis=1) == y_val)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        tqdm.write(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

def hparam_search():
    wandb.init()
    config = wandb.config

    epochs = config["epochs"]
    num_layers = config["num_hidden_layers"] + 1
    hidden_layer_size = config["hidden_layer_size"]
    layer_dims = [X_train.shape[1]] + [hidden_layer_size] * (num_layers - 1) + [10]
    activation = config["activation"]
    activations = [activation] * (num_layers - 1) + ["softmax"]
    
    weight_init = config["weight_init"]

    nn = NeuralNetwork(num_layers, layer_dims, activations, weight_init)

    optimizer = OptimizerFactory.get_optimizer(
        config["optimizer"], nn, 
        learning_rate=config["learning_rate"],
        momentum=config.get("momentum", 0.9), 
        beta1=config.get("beta1", 0.9), 
        beta2=config.get("beta2", 0.999), 
        epsilon=config.get("epsilon", 1e-8)
    )

    train(nn, X_train, y_train, X_val, y_val, optimizer, epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network with hyperparameter optimization using wandb.")
    parser.add_argument("-wp", "--wandb_project", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=None, help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function to use")
    parser.add_argument("-o", "--optimizer", default=None, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=None, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=None, help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=None, help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=None, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=None, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=None, help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=None, help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", default=None, choices=["random", "xavier"], help="Weight initialization method")
    parser.add_argument("-nhl", "--num_hidden_layers", type=int, default=None, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_layer_size", type=int, default=None, help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", default=None, choices=["identity", "sigmoid", "tanh", "ReLU"], help="Activation function to use")
    parser.add_argument("-c", "--count", type=int, default=25, help="Number of sweep runs")


    args = parser.parse_args()

    params = ["epochs", "num_hidden_layers", "hidden_layer_size", "weight_decay", 
              "learning_rate", "beta1", "beta2", "momentum", "epsilon", 
              "optimizer", "batch_size", "weight_init", "activation"]

    for param in params:
        value = getattr(args, param)
        if value:
            sweep_config["parameters"][param]["values"] = [value]

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
    X_train, X_val = X_train[:-6000], X_train[-6000:]
    y_train, y_val = y_train[:-6000], y_train[-6000:]

    X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0
    
    sweep_kwargs = {}
    if args.wandb_project:
        sweep_kwargs["project"] = args.wandb_project
    if args.wandb_entity:
        sweep_kwargs["entity"] = args.wandb_entity
    
    sweep_id = wandb.sweep(sweep_config, **sweep_kwargs)

    wandb.agent(sweep_id, function=hparam_search, count=args.count)