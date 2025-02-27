import numpy as np

class Neuron:
    def __init__(self, w, b):
        self.w = np.array(w)
        self.b = np.array(b)
        self.a = None
        self.h = None
    
    def pre_activation(self, h):
        return np.dot(self.w, h) + self.b
    
    def sigmoid(self, pre_activated):
        return 1 / (1 + np.exp(-pre_activated))

    def tanh(self, pre_activated):
        return np.tanh(pre_activated)
    
    def relu(self, pre_activated):
        return np.maximum(0, pre_activated)
    
    def activation(self, a, activation='sigmoid'):
        activation_options = {
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'relu': self.relu
        }
        return activation_options[activation](a)
    
    def feedforward(self, x):
        self.a = self.pre_activation(x)
        self.h = self.activation(self.a)
    
    def __repr__(self):
        return f'Neuron({self.w.shape}, {self.b.shape})'
    