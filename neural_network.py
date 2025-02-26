import numpy as np
from neuron import Neuron

class NeuralNetwork:
    def __init__(self, layers, neurons_per_layer):
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.neurons = {
            0: [
                Neuron(
                    w=np.random.rand(neurons_per_layer[0]),
                    b=np.random.rand(1)
                )
            for _ in range(neurons_per_layer[0])],
            **{
                layer: [
                    Neuron(
                        w=np.random.rand(neurons_per_layer[layer-1]),
                        b=np.random.rand(1)
                    ) 
                for _ in range(neurons_per_layer[layer])]
            for layer in range(1, layers)},
        }
        
    def feedforward(self, x):
        x = np.array(x).reshape(-1,)
        h_current = np.array([neuron.feedforward(x) for neuron in self.neurons[0]])
        for layer in range(1, self.layers):
            h_previous = h_current
            h_current = np.array([neuron.feedforward(h_previous) for neuron in self.neurons[layer]])
        return h_current
    
    def __repr__(self):
        return f'''NeuralNetwork(Layers = {self.layers}, Neurons per layer = {self.neurons_per_layer})'''