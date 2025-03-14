import numpy as np
from losses import LossFnFactory
from activations import ActivationFnFactory

np.random.seed(42)

class NeuralNetwork:    
    def __init__(self, 
                 num_layers, neurons_per_layer, 
                 activation="relu", weight_init="xavier", loss="cross_entropy",
                 learning_rate=0.001, weight_decay=0.0, batch_size=1):
        if len(neurons_per_layer) != num_layers + 1:
            raise ValueError("Invalid configuration: num_layers and neurons_per_layer mismatch")

        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer

        self.activation_fn = ActivationFnFactory.get(activation)
        self.loss_fn = LossFnFactory.get(loss)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.weights, self.biases = self.init_params(weight_init, neurons_per_layer)
        

    def init_params(self, init_method, neurons_per_layer):
        weights, biases = [], []

        for i in range(self.num_layers):
            W = np.random.randn(neurons_per_layer[i], neurons_per_layer[i+1], )
            if init_method == "xavier":
                W *= np.sqrt(1 / neurons_per_layer[i])

            weights.append(W)
            biases.append(np.zeros((1, neurons_per_layer[i+1])))

        return weights, biases

    def forward(self, X):
        self.activations = [X]
        self.pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            a = np.dot(self.activations[-1], W) + b
            self.pre_activations.append(a)

            if i == self.num_layers - 1:
                if self.loss_fn.__class__.__name__ == "CrossEntropyLoss":
                    h = ActivationFnFactory.get("softmax").activation(a)
                else:
                    h = ActivationFnFactory.get("identity").activation(a)
            else:
                h = self.activation_fn.activation(a)
            self.activations.append(h)

        return self.activations[-1]

    def backward(self, Y):
        m = Y.shape[0]
        dW, db = [], []

        dH = self.loss_fn.derivative(y_true=Y, y_pred=self.activations[-1])

        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                dA = dH
            else:
                dA = dH * self.activation_fn.derivative(self.pre_activations[i])
            
            dW.insert(0, np.dot(self.activations[i].T, dA) / m)
            db.insert(0, np.sum(dA, axis=0, keepdims=True) / m)
            
            dH = np.dot(dA, self.weights[i].T)

        return dW, db
    
    def __repr__(self):
        layers_info = "\n".join([f"Layer {i}:\n\tInput dim: {self.neurons_per_layer[i]}\n\tOutput dim: {self.neurons_per_layer[i+1]}" 
                                 for i in range(self.num_layers)])
        return f"""Neural Network:\nNo. of Layers: {self.num_layers}\n{layers_info}"""