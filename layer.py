import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation

    def activation_fn(self, a):
        if self.activation == 'sigmoid':
            return np.where(
                a >= 0,
                1 / (1 + np.exp(-a)),
                np.exp(a) / (1 + np.exp(a))
            )
        elif self.activation == 'tanh':
            return np.tanh(a)
        elif self.activation == 'relu':
            return np.maximum(0, a)
        elif self.activation == 'softmax':
            h = np.exp(a - np.max(a))
            return h / np.sum(h, axis=1, keepdims=True)
        elif self.activation == 'linear':
            return a
        else: 
            raise ValueError(f'Activation function {self.activation} not supported')
    
    def forward(self, X):
        X = X.reshape(1, -1)
        self.a = np.dot(X, self.weights) + self.biases
        self.h = self.activation_fn(self.a)
        return self.h
    
    def __repr__(self):
        return f"""\tInput dim:{self.weights.shape[0]} \n\tOutput dim:{self.weights.shape[1]} \n\tActivation:{self.activation}"""
    
    def activation_fn_prime(self, a):
        if self.activation == 'sigmoid':
            return self.h * (1 - self.h)
        elif self.activation == 'tanh':
            return 1 - np.power(self.h, 2)
        elif self.activation == 'relu':
            return np.where(self.a >= 0, 1, 0)
        elif self.activation == 'softmax':
            return 1
        elif self.activation == 'linear':
            return 1
        else: 
            raise ValueError(f'Activation function {self.activation} not supported')
        
    def backward(self, dH, h_minus):
        dA = dH * self.activation_fn_prime(self.a)
        
        dW = np.dot(h_minus.T, dA)
        db = np.sum(dA, axis=0, keepdims=True)

        dH_minus = np.dot(dA, self.weights.T)
        
        return dW, db, dH_minus