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
        print(f"Layer 0 output shape: {h_current.shape}")
        for layer in range(1, self.layers):
            h_previous = h_current
            h_current = np.array([neuron.feedforward(h_previous) for neuron in self.neurons[layer]])
            print(f"Layer {layer} output shape: {h_current.shape}")
        return h_current
    
    def __repr__(self):
        return f'''NeuralNetwork({self.layers}, {self.neurons_per_layer})'''