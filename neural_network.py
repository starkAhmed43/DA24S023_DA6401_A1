import numpy as np
from neuron import Neuron

class NeuralNetwork:
    def __init__(self, layers, neurons_per_layer, activations):
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.activations = {0: None, **{i: activations[i-1] for i in range(1, layers)}}

        self.weights = {i: np.random.randn(neurons_per_layer[i], neurons_per_layer[i-1]) / np.sqrt(neurons_per_layer[i-1]) for i in range(1, layers)}

        self.biases = {i: np.zeros((neurons_per_layer[i], 1)) for i in range(1, layers)}

    def activation_fn(self, a, activation):
        if activation == 'sigmoid':
            return np.where(
                a >= 0,
                1 / (1 + np.exp(-a)),
                np.exp(a) / (1 + np.exp(a))
            )
        elif activation == 'tanh':
            return np.tanh(a)
        elif activation == 'relu':
            return np.maximum(0, a)
        elif activation == 'softmax':
            h = np.exp(a - np.max(a))
            return h / np.sum(h, axis=0, keepdims=True)
        return a

    def activation_fn_prime(self, a, activation):
        h = self.activation_fn(a, activation)
        if activation == 'sigmoid':
            return h * (1 - h)
        elif activation == 'tanh':
            return 1 - h**2
        elif activation == 'relu':
            return (h > 0).astype(float)
        else:
            return np.ones_like(a)
        

    def feedforward(self, x):
        h = np.array(x).reshape(-1, 1)
        self.activations_cache = {0: h}
        self.pre_activation_cache = {0: np.array(None)}

        for i in range(1, self.layers):
            a = self.weights[i] @ h + self.biases[i]
            h = self.activation_fn(a, self.activations[i])

            self.pre_activation_cache[i] = a
            self.activations_cache[i] = h
        return h
    
    def __repr__(self):
        return f'''NeuralNetwork(
            Layers: {self.layers}, 
            Neurons per layer: {self.neurons_per_layer}, 
            Activations: {self.activations},
            Weights shape: { {k: v.shape for k, v in self.weights.items()} }, 
            Biases shape: { {k: v.shape for k, v in self.biases.items()} },
            Pre-activations shape: { {k: v.shape for k, v in self.pre_activation_cache.items()} }, 
            Activations shape: { {k: v.shape for k, v in self.activations_cache.items()} }
        )'''
    
    def backprop(self, y_hat, y_class_label):
        y_true = np.zeros_like(y_hat)
        y_true[y_class_label] = 1

        delta_W = {i: np.zeros_like(self.weights[i]) for i in range(1, self.layers)}
        delta_b = {i: np.zeros_like(self.biases[i]) for i in range(1, self.layers)}

        dL_dh = np.array(None)
        dL_da = y_hat - y_true

        for i in range(self.layers-1, 0, -1):
            print(f'Layer {i}')

            print(f'delta_W[{i}] shape: {delta_W[i].shape}')
            del_W = dL_da @ self.activations_cache[i-1].T
            print(del_W)
            delta_W[i] = dL_da @ self.activations_cache[i-1].T
            delta_b[i] = dL_da

            if i > 1:
                dL_dh = self.weights[i].T @ dL_da
                dL_da = dL_dh * self.activation_fn_prime(self.pre_activation_cache[i-1], self.activations[i-1])
        
        return delta_W, delta_b, dL_da, dL_dh