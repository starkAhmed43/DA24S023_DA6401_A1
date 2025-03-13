import numpy as np

class Sigmoid:
    def activation(self, x):
        return 1 / (1 + np.exp(-x))
    def derivative(self, x):
        return self.activation(x) * (1 - self.activation(x))
    
class Tanh:
    def activation(self, x):
        return np.tanh(x)
    def derivative(self, x):
        return 1 - np.tanh(x)**2

class ReLU:
    def activation(self, x):
        return np.maximum(0, x)
    def derivative(self, x):
        return (x>0).astype(float)

class Identity:
    def activation(self, x):
        return x
    def derivative(self, x):
        return 1

class Softmax:
    def activation(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    def derivative(self, x):
        return 1

class ActivationFnFactory:
    ACTIVATIONS = {
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'relu': ReLU,
        'identity': Identity,
        'softmax': Softmax
    }

    @staticmethod
    def get(activation):
        activation = activation.lower()
        if activation not in ActivationFnFactory.ACTIVATIONS:
            print(f"Activation function {activation} not supported. Defaulting to ReLU.")
            activation = 'relu'
        return ActivationFnFactory.ACTIVATIONS[activation]()