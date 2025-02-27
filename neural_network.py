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
        self.all_w_list, self.all_b_list = {layer: None for layer in range(layers)}, {layer: None for layer in range(layers)}
        self.all_a_list, self.all_h_list = {layer: None for layer in range(layers)}, {layer: None for layer in range(layers)}
        self.all_delta_a_list, self.all_delta_h_list = {layer: None for layer in range(layers)}, {layer: None for layer in range(layers)}
        self.all_delta_w_list, self.all_delta_b_list = {layer: None for layer in range(layers)}, {layer: None for layer in range(layers)}
        self.yhat = None
        
    def feedforward(self, x):
        x = np.array(x).reshape(-1,)

        # feedforward for the first / input layer
        a_layerwise, h_layerwise, w_layerwise, b_layerwise = [], [], [], []
        for neuron in self.neurons[0]:
            neuron.feedforward(x)
            a_layerwise.append(neuron.a)
            h_layerwise.append(neuron.h)
            w_layerwise.append(neuron.w)
            b_layerwise.append(neuron.b)
        self.all_a_list[0] = np.array(a_layerwise)
        self.all_h_list[0] = np.array(h_layerwise)
        self.all_w_list[0] = np.array(w_layerwise)
        self.all_b_list[0] = np.array(b_layerwise)

        # feedforward for the rest of the layers (hidden and output)
        for layer in range(1, self.layers):
            a_layerwise, h_layerwise, w_layerwise, b_layerwise = [], [], [], []
            for neuron in self.neurons[layer]:
                neuron.feedforward(self.all_h_list[layer-1])
                a_layerwise.append(neuron.a)
                h_layerwise.append(neuron.h)
                w_layerwise.append(neuron.w)
                b_layerwise.append(neuron.b)
            self.all_a_list[layer] = np.array(a_layerwise)
            self.all_h_list[layer] = np.array(h_layerwise)
            self.all_w_list[layer] = np.array(w_layerwise)
            self.all_b_list[layer] = np.array(b_layerwise)
        
        h_final = self.all_h_list[self.layers-1]
        h_final_exp_sum = np.sum(np.exp(h_final))
        self.y_hat = np.exp(h_final) / h_final_exp_sum

        return self.y_hat
    

    def cross_entropy_loss(self, class_label):
        return -np.log(self.y_hat[class_label])
    
    # gradient wrt to h
    def gradient_wrt_h(self, w_next, gradient_a_next):
        return np.dot(w_next.T, gradient_a_next)

    # gradient wrt to a
    def gradient_wrt_a(self, gradient_h, a):
        def sigmoid_prime(a):
            return a * (1 - a)
        
        return gradient_h * sigmoid_prime(a)

    # gradient wrt to weights
    def gradient_wrt_w(self, gradient_a, h):
        return np.outer(gradient_a, h.T)

    # gradient wrt to bias
    def gradient_wrt_b(self, gradient_a):
        return gradient_a
    

    # gradient wrt to outer layer
    def gradient_wrt_output(self, class_label):
        a_delta = self.y_hat.copy()
        a_delta[class_label] -= 1
        self.all_delta_a_list[self.layers-1] = a_delta
        
        b_delta = self.gradient_wrt_b(a_delta)
        self.all_delta_b_list[self.layers-1] = b_delta

        w_delta = self.gradient_wrt_w(a_delta, self.all_h_list[self.layers-1])   
        self.all_delta_w_list[self.layers-1] = w_delta

    # gradient wrt to hidden layers and input layer
    def gradient_wrt_hidden(self):
        for layer in range(self.layers-2, 0, -1):
            h_delta = self.gradient_wrt_h(self.all_w_list[layer+1], self.all_delta_a_list[layer+1])
            self.all_delta_h_list[layer] = h_delta

            a_delta = self.gradient_wrt_a(h_delta, self.all_a_list[layer])
            self.all_delta_a_list[layer] = a_delta

            b_delta = self.gradient_wrt_b(a_delta)
            self.all_delta_b_list[layer] = b_delta

            w_delta = self.gradient_wrt_w(a_delta, self.all_h_list[layer-1])
            self.all_delta_w_list[layer] = w_delta
    
    # backpropagation
    def backprop(self, class_label):
        self.gradient_wrt_output(class_label)
        self.gradient_wrt_hidden()

    def __repr__(self):
        return f'''NeuralNetwork(Layers = {self.layers}, Neurons per layer = {self.neurons_per_layer})'''