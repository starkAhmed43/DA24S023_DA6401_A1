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
        all_a_list, all_h_list = [], []
        a_layerwise, h_layerwise = [], []

        for neuron in self.neurons[0]:
            a, h = neuron.feedforward(x)
            a_layerwise.append(a)
            h_layerwise.append(h)
        all_a_list.append(np.array(a_layerwise))
        all_h_list.append(np.array(h_layerwise))

        for layer in range(1, self.layers):
            a_layerwise, h_layerwise = [], []
            for neuron in self.neurons[layer]:
                a, h = neuron.feedforward(all_h_list[layer-1])
                a_layerwise.append(a)
                h_layerwise.append(h)
            all_a_list.append(np.array(a_layerwise))
            all_h_list.append(np.array(h_layerwise))
        
        h_final = all_h_list[-1]
        h_final_exp_sum = np.sum(np.exp(h_final))
        y_hat = np.exp(h_final) / h_final_exp_sum
        return all_a_list, all_h_list[:-1], y_hat
    
    def __repr__(self):
        return f'''NeuralNetwork(Layers = {self.layers}, Neurons per layer = {self.neurons_per_layer})'''