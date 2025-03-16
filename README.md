# DA6401-A1

This project implements a feedforward neural network to classify images from the Fashion-MNIST dataset or the MNIST dataset. The code is designed to be flexible, allowing easy changes to the number of hidden layers and the number of neurons in each hidden layer.

# WANDB Report Link:

https://wandb.ai/starkahmed43/DA24S023_DA6401_A1/reports/DA6401-Assignment-1--VmlldzoxMTgwMTI1OQ

# GitHub Repo Link:

https://github.com/starkAhmed43/DA24S023_DA6401_A1.git

## Project Structure

```
.gitignore
activations.py
best_model.py
dataloader.py
losses.py
mnist_sweep.py
neural_network.py
optimizers.py
README.md
requirements.txt
train.py
```

## Files

- `activations.py`: Contains activation functions used in the neural network.
- `best_model.py`: Script to save and load the best-performing model.
- `dataloader.py`: Handles loading and preprocessing of datasets.
- `losses.py`: Contains loss functions used for training the neural network.
- `mnist_sweep.py`: Sweeps MNIST across 3 hparam configs.
- `neural_network.py`: Contains the implementation of the neural network.
- `optimizers.py`: Contains various optimizer implementations.
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

## Training

The training function is defined in [`train.py`](train.py):

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

## Self Declaration
I, Adhil Ahmed P M Shums (DA24S023), swear on my honour that I have written the code and the report by myself and have not copied it from the internet or other students.