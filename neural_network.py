import numpy as np
from layer import Layer

class NeuralNetwork:
    def __init__(self, num_layers, neurons_per_layer, activations):
        if len(neurons_per_layer) != num_layers + 1 or len(activations) != num_layers:
            raise ValueError("Invalid configuration: Check number of layers, neurons, and activations.")

        self.layers = []
        for i in range(num_layers):
            layer = Layer(neurons_per_layer[i], neurons_per_layer[i + 1], activations[i])
            self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, X, y_hat_probs, y_true_class):
        X = X.reshape(1, -1)
        y_true_one_hot = np.zeros_like(y_hat_probs)
        y_true_one_hot[0, y_true_class] = 1
        
        # gradients wrt output layer
        dA_L = y_hat_probs - y_true_one_hot
        
        all_delta_W = [np.dot(self.layers[-2].h.T, dA_L)]
        all_delta_b = [np.sum(dA_L, axis=0, keepdims=True)]
        
        ### find the gradient of loss wrt to activations of the last hidden layer
        ### and then use it to find the gradients wrt to weights and biases of all layers
        dH_minus = np.dot(dA_L, self.layers[-1].weights.T)


        # gradient wrt hidden layers EXCEPT THE FIRST HIDDEN LAYER
        for l_idx in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[l_idx]
            prev_layer = self.layers[l_idx - 1]
            
            ### the dH_minus being passed to layer.backward() is from the previous iteration
            ### so in the current iteration, it is the dH of the current layer
            
            ### h_minus being passed to layer.backward() in the current iteration refers to the h of the previous layer
            ### i.e if the current iteration is for the 4th layer, we send in the activations of the 3rd layer
            dW, db, dH_minus = layer.backward(dH=dH_minus, h_minus=prev_layer.h)

            ### layer.backward() will then return the dW and db for the current layer and the dH for the previous layer
            ### i.e if the current iteration is for the 4th layer, we get the dW and db for the 4th layer and the dH for the 3rd layer

            all_delta_W.insert(0, dW)
            all_delta_b.insert(0, db)
        

        # gradient wrt first hidden layer

        ### here dH_minus is the one returned by the backward() of the 2nd hidden layer
        ### thus, it is the very own dH of the first hidden layer
        ### and h_minus is the input X since there are no previous layers
        dW_0, db_0, _ = self.layers[0].backward(dH=dH_minus, h_minus=X)
        
        all_delta_W.insert(0, dW_0)
        all_delta_b.insert(0, db_0)
        return all_delta_W, all_delta_b
    
    def __repr__(self):
        layers_repr = "\n".join([f"Layer {i}:\n {layer}" for i, layer in enumerate(self.layers)])
        return f"""Neural Network: \nNo. of Layers: {len(self.layers)} \nLayers: \n{layers_repr}"""