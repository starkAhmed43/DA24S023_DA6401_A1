# DA6401-A1

This project implements a feedforward neural network to classify images from the Fashion-MNIST dataset or the MNIST dataset. The code is designed to be flexible, allowing easy changes to the number of hidden layers and the number of neurons in each hidden layer.

## Project Structure

```
.gitignore
A1.ipynb
layer.py
neural_network.py
optimizer.py
requirements.txt
train.py
```

## Files

- `A1.ipynb`: Jupyter notebook that contains implementation of Q1 - Q3.
- `layer.py`: Contains the implementation of a neural network layer.
- `neural_network.py`: Contains the implementation of the neural network.
- `optimizer.py`: Contains various optimizer implementations.
- `requirements.txt`: Lists the dependencies required for the project.
- `train.py`: Contains the training loop for the neural network.

## Optimizers

The following optimizers are implemented in [`optimizer.py`](optimizer.py):

- `SGDOptimizer`: Standard Stochastic Gradient Descent.
- `MomentumGDOptimizer`: Gradient Descent with Momentum.
- `NAGOptimizer`: Nesterov Accelerated Gradient.
- `RMSPropOptimizer`: Root Mean Square Propagation.
- `AdamOptimizer`: Adaptive Moment Estimation.
- `NadamOptimizer`: Nesterov-accelerated Adaptive Moment Estimation.

### Example Usage

```python
from neural_network import NeuralNetwork
from optimizer import OptimizerFactory

layers = np.random.randint(4, 10)
neurons_in_input_layer = x_train[0].reshape(-1).shape[0]
neurons_per_hidden_layer = [np.random.randint(3, 10) for _ in range(layers - 1)]
neurons_in_output_layer = 10
neurons_per_layer = [neurons_in_input_layer] + neurons_per_hidden_layer + [neurons_in_output_layer]

activation_options = ["relu", "sigmoid", "tanh", "linear"]
activations = [activation_options[np.random.randint(0, 4)] for _ in range(layers - 1)]
activations.append("softmax")

model = NeuralNetwork(layers, neurons_per_layer, activations)

optimizer = OptimizerFactory.get_optimizer("adam", model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
```

## Training

The training function is defined in [`train.py`](train.py):

```python
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
```

## Arguments

The script [`train.py`](train.py) accepts the following command-line arguments:

- `-wp`, `--wandb_project`: Project name used to track experiments in Weights & Biases dashboard.
- `-we`, `--wandb_entity`: Wandb Entity used to track experiments in the Weights & Biases dashboard.
- `-d`, `--dataset`: Dataset to use (`mnist` or `fashion_mnist`).
- `-e`, `--epochs`: Number of epochs to train the neural network.
- `-b`, `--batch_size`: Batch size used to train the neural network.
- `-l`, `--loss`: Loss function to use (`mean_squared_error` or `cross_entropy`).
- `-o`, `--optimizer`: Optimizer to use (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`).
- `-lr`, `--learning_rate`: Learning rate used to optimize model parameters.
- `-m`, `--momentum`: Momentum used by momentum and nag optimizers.
- `-beta`, `--beta`: Beta used by rmsprop optimizer.
- `-beta1`, `--beta1`: Beta1 used by adam and nadam optimizers.
- `-beta2`, `--beta2`: Beta2 used by adam and nadam optimizers.
- `-eps`, `--epsilon`: Epsilon used by optimizers.
- `-w_d`, `--weight_decay`: Weight decay used by optimizers.
- `-w_i`, `--weight_init`: Weight initialization method (`random` or `xavier`).
- `-nhl`, `--num_hidden_layers`: Number of hidden layers used in the feedforward neural network.
- `-sz`, `--hidden_layer_size`: Number of hidden neurons in a feedforward layer.
- `-a`, `--activation`: Activation function to use (`identity`, `sigmoid`, `tanh`, `ReLU`).
- `-c`, `--count`: Number of sweep runs.

The params specified in the command-line argumnets override the corresponding range defined in the `wandb` sweep config. 

Thus to run a hparam sweep across all the params, DO NOT specify any params via args. 

If the `wandb` project details are not specified via args, the default values are used assuming `wandb login` has already been done. 

If the `-d`, `--dataset` flag is not specifed it will default to `fashion_mnist`.

### Example Usage

```sh
python train.py -wp my_project -we my_entity -d fashion_mnist -e 100 -b 32 -l cross_entropy -o adam -lr 0.001 -beta1 0.9 -beta2 0.999 -eps 1e-8 -w_i xavier -nhl 3 -sz 64 -a relu -c 25
```

## Logging

The project uses Weights & Biases (wandb) for logging. Logs can be found in the `wandb/` directory.