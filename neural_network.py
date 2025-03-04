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

    def activation_fn(self, z, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'softmax':
            a = np.exp(z - np.max(z))
            return a / np.sum(a, axis=0, keepdims=True)
        return z

    def activation_fn_prime(self, a, activation):
        if activation == 'sigmoid':
            return a * (1 - a)
        elif activation == 'tanh':
            return 1 - a**2
        elif activation == 'relu':
            return (a > 0).astype(float)
        else:
            return a
        

    def feedforward(self, x):
        h = np.array(x).reshape(-1, 1)
        self.activations_cache = [h]  
        self.pre_activation_cache = []

        for i in range(self.layers - 1):
            z = self.weights[i] @ h + self.biases[i]
            h = self.activation_fn(z, self.activations[i])

            self.pre_activation_cache.append(z)
            self.activations_cache.append(h)
        return h
    
    def __repr__(self):
        return f'NeuralNetwork({self.layers}, {self.neurons_per_layer}, {self.activations})'
