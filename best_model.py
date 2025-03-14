import wandb
import argparse
import numpy as np
from train import train
from tqdm.auto import tqdm
from dataloader import load_data
from optimizers import OptimizerFactory
from neural_network import NeuralNetwork

np.random.seed(42)
class_labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

def log_class_samples(X, y, dataset_name):
    class_labels_dict = {
        "mnist": [str(i) for i in range(10)],  # Digits 0-9
        "fashion_mnist": class_labels
    }

    class_names = class_labels_dict[dataset_name]
    unique_classes = np.unique(np.argmax(y, axis=1))  # Get unique class indices

    for cls in unique_classes:
        idx = np.where(np.argmax(y, axis=1) == cls)[0][0]  # Get first index of class
        img = X[idx].reshape(28, 28)  # Reshape assuming 28x28 image

        # Log each image separately to W&B
        wandb.log({
            f"Class {class_names[cls]}": wandb.Image(img, caption=class_names[cls])
        }, step=0)

def get_best_params(wandb_entity, project_name, dataset_name):
    api = wandb.Api()
    sweeps = api.project(name=project_name, entity=wandb_entity).sweeps()
    tqdm.write(f"Found {len(sweeps)} sweeps in project '{project_name}'.")

    best_run = None
    best_val_accuracy = -float("inf")

    for sweep in sweeps:
        for run in tqdm(sweep.runs, desc=f"Searching for best run in sweep {sweep.id}", unit="run"):
            if run.config["dataset"] != dataset_name:
                continue
            val_accuracy = run.summary.get("val_accuracy", -float("inf"))
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_run = run

    if best_run is None:
        raise ValueError(f"No runs found for dataset '{dataset_name}' in project '{project_name}'.")

    tqdm.write(f"Best run found with validation accuracy: {best_val_accuracy:.4f}")
    return best_run.config

def init_model(X_train, config):
    num_layers = config["num_hidden_layers"] + 1
    hidden_layer_size = config["hidden_layer_size"]
    layer_dims = [X_train.shape[1]] + [hidden_layer_size] * (num_layers - 1) + [10]

    model = NeuralNetwork(
        num_layers=num_layers,
        neurons_per_layer=layer_dims,
        activation=config["activation"],
        weight_init=config["weight_init"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
    )

    optimizer = OptimizerFactory.get_optimizer(
        name=config["optimizer"],
        model=model,
        weight_decay=config["weight_decay"],
        learning_rate=config["learning_rate"],
        momentum=config["momentum"],
        beta1=config["beta1"],
        beta2=config["beta2"],
        epsilon=config["epsilon"]
    )

    return model, optimizer

def evaluate_model(nn, X, y):
    probs = nn.forward(X)
    loss_fn = nn.loss_fn.loss
    loss = loss_fn(y, probs)
    accuracy = np.mean(np.argmax(probs, axis=1) == np.argmax(y, axis=1))
    return loss, accuracy, probs

def plot_and_log_confusion_matrix(y_test, test_probs, class_labels):
    wandb.log({
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            title="Test Confusion Matrix",
            probs=test_probs,
            y_true=np.argmax(y_test, axis=1),
            class_names=class_labels
        )
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the best model using W&B best parameters.")
    parser.add_argument("-we", "--wandb_entity", required=True, help="W&B entity name.")
    parser.add_argument("-wp", "--project_name", required=True, help="W&B project name.")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name (e.g., 'mnist', 'fashion_mnist').")
    args = parser.parse_args()

    # Get the best parameters from W&B
    best_params = get_best_params(args.wandb_entity, args.project_name, args.dataset)

    wandb.init(project=args.project_name, name=f"best_model_{args.dataset}")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)
    log_class_samples(X_train, y_train, args.dataset) 

    # Initialize the model and optimizer
    model, optimizer = init_model(X_train, best_params)

    # Train the model
    train(model=model,
          X_train=X_train,
          y_train=y_train,
          X_val=X_val,
          y_val=y_val,
          optimizer=optimizer,
          epochs=best_params["epochs"],
          batch_size=best_params["batch_size"])

    # Evaluate the model
    train_loss, train_accuracy, _ = evaluate_model(model, X_train, y_train)
    val_loss, val_accuracy, _ = evaluate_model(model, X_val, y_val)
    test_loss, test_accuracy, test_probs = evaluate_model(model, X_test, y_test)

    # Print results
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    wandb.config.update(best_params)
    plot_and_log_confusion_matrix(y_test, test_probs, class_labels)
    wandb.finish()