import numpy as np
from neuron import Neuron

class NeuralNetwork:
    def __init__(self, layers, neurons_per_layer, activations):
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.activations = activations

        self.weights = [np.random.randn(neurons_per_layer[i+1], neurons_per_layer[i]) * np.sqrt(2.0 / neurons_per_layer[i])
                        for i in range(layers - 1)]
        self.biases = [np.zeros((neurons_per_layer[i+1], 1)) for i in range(layers - 1)]

    def activation_fn(self, a, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-a))
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
            return (a > 0).astype(float)
        else:
            return np.ones_like(a)
        

    def feedforward(self, x):
        h = np.array(x).reshape(-1, 1)
        self.activations_cache = [h]  
        self.pre_activation_cache = []

        for i in range(self.layers - 1):
            a = self.weights[i] @ h + self.biases[i]
            h = self.activation_fn(a, self.activations[i])

            self.pre_activation_cache.append(a)
            self.activations_cache.append(h)
        return h
    
    def __repr__(self):
        return f'NeuralNetwork({self.layers}, {self.neurons_per_layer}, {self.activations})'
