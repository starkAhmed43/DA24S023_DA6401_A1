import numpy as np
from tqdm.auto import tqdm

def momentum_gd(epochs, eta, beta, X, Y, nn, clip_value=5.0):
    prev_uw, prev_ub = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}

    for epoch in tqdm(range(epochs), desc="Epochs"):
        delta_w, delta_b = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}
        for x, y in tqdm(zip(X, Y), desc="Samples", total=len(X), leave=False):
            y_hat = nn.feedforward(x)
            w_list, b_list = nn.all_w_list, nn.all_b_list
            
            d_w, d_b = nn.backprop(w_list, b_list, y_hat, y)
            for layer in range(1, nn.layers):
                delta_w[layer] += d_w[layer]
                delta_b[layer] += d_b[layer]
        
        # Gradient clipping
        for layer in range(1, nn.layers):
            delta_w[layer] = np.clip(delta_w[layer], -clip_value, clip_value)
            delta_b[layer] = np.clip(delta_b[layer], -clip_value, clip_value)
        

        uw, ub = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}
        for layer in range(1, nn.layers):
            uw[layer] = (beta * prev_uw[layer]) + (eta * delta_w[layer])
            ub[layer] = (beta * prev_ub[layer]) + (eta * delta_b[layer])

        nn.update_weights(uw, ub)
        prev_uw, prev_ub = uw, ub
            