import numpy as np
from tqdm.auto import tqdm

def make_batches(batch_size, X, Y):
    X_batches, Y_batches = [], []
    n_batches = len(X) // batch_size
    remainder = len(X) % n_batches
    start = 0
    for i in range(n_batches):
        end = start + batch_size + (1 if i < remainder else 0)
        X_batches.append(np.array(X[start:end]))
        Y_batches.append(np.array(Y[start:end]))
        start = end
    return X_batches, Y_batches

def momentum_gd(epochs, eta, beta, X, Y, nn, clip_value=5.0, batch_size=-1):
    if batch_size == -1:
        batch_size = len(X)
    X_batches, Y_batches = make_batches(batch_size, X, Y)

    prev_uw, prev_ub = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}

    for epoch in tqdm(range(epochs), desc="Epochs"):
        w_list, b_list = nn.all_w_list, nn.all_b_list

        for X_batch, Y_batch in tqdm(zip(X_batches, Y_batches), desc="Batches", total=len(X_batches), leave=False):
            delta_w, delta_b = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}
            for x, y in zip(X_batch, Y_batch):
                y_hat = nn.feedforward(x)
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

def nestorov_acc_gd(epochs, eta, beta, X, Y, nn, clip_value=5.0, batch_size=-1):
    if batch_size == -1:
        batch_size = len(X)
    X_batches, Y_batches = make_batches(batch_size, X, Y)

    prev_uw, prev_ub = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}

    for epoch in tqdm(range(epochs), desc="Epochs"):
        v_w, v_b = {layer: beta* prev_uw[layer] for layer in range(1, nn.layers)}, {layer: beta* prev_ub[layer] for layer in range(1, nn.layers)}
        
        w_list, b_list = nn.all_w_list, nn.all_b_list
        w_list, b_list = {layer: w_list[layer] - v_w[layer] for layer in range(1, nn.layers)}, {layer: b_list[layer] - v_b[layer] for layer in range(1, nn.layers)}

        for X_batch, Y_batch in tqdm(zip(X_batches, Y_batches), desc="Batches", total=len(X_batches), leave=False):
            delta_w, delta_b = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}
            for x, y in zip(X_batch, Y_batch):
                y_hat = nn.feedforward(x)
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
            
def sgd(epochs, eta, X, Y, nn, clip_value=5.0, batch_size=1):
    X_batches, Y_batches = make_batches(batch_size, X, Y)

    w_list, b_list = nn.all_w_list, nn.all_b_list

    for epoch in tqdm(range(epochs), desc="Epochs"):
        delta_w, delta_b = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}
        
        for X_batch, Y_batch in tqdm(zip(X_batches, Y_batches), desc="Batches", total=len(X_batches), leave=False):
            for x, y in zip(X_batch, Y_batch):
                y_hat = nn.feedforward(x)
                d_w, d_b = nn.backprop(w_list, b_list, y_hat, y)
                for layer in range(1, nn.layers):
                    delta_w[layer] += d_w[layer]
                    delta_b[layer] += d_b[layer]
            
            # Gradient clipping
            for layer in range(1, nn.layers):
                delta_w[layer] = eta * np.clip(delta_w[layer], -clip_value, clip_value)
                delta_b[layer] = eta * np.clip(delta_b[layer], -clip_value, clip_value)
            
            nn.update_weights(delta_w, delta_b)

def rmsprop(epochs, eta, beta, X, Y, nn, clip_value=5.0, batch_size=16):
    X_batches, Y_batches = make_batches(batch_size, X, Y)

    uw, ub = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}
    epsilon = 1e-8

    for epoch in tqdm(range(epochs), desc="Epochs"):
        w_list, b_list = nn.all_w_list, nn.all_b_list
        delta_w, delta_b = {layer: 0 for layer in range(1, nn.layers)}, {layer: 0 for layer in range(1, nn.layers)}

        for X_batch, Y_batch in tqdm(zip(X_batches, Y_batches), desc="Batches", total=len(X_batches), leave=False):
            for x, y in zip(X_batch, Y_batch):
                y_hat = nn.feedforward(x)
                d_w, d_b = nn.backprop(w_list, b_list, y_hat, y)
                for layer in range(1, nn.layers):
                    delta_w[layer] += d_w[layer]
                    delta_b[layer] += d_b[layer]

            # Gradient clipping
            for layer in range(1, nn.layers):
                delta_w[layer] = np.clip(delta_w[layer], -clip_value, clip_value)
                delta_b[layer] = np.clip(delta_b[layer], -clip_value, clip_value)

            uw = {layer: (beta * uw[layer]) + ((1 - beta) * (delta_w[layer] ** 2)) for layer in range(1, nn.layers)}
            ub = {layer: (beta * ub[layer]) + ((1 - beta) * (delta_b[layer] ** 2)) for layer in range(1, nn.layers)}

            delta_w = {layer: (eta * delta_w[layer] / np.sqrt(uw[layer] + epsilon))  for layer in range(1, nn.layers)}
            delta_b = {layer: (eta * delta_b[layer] / np.sqrt(ub[layer] + epsilon))  for layer in range(1, nn.layers)}

            nn.update_weights(delta_w, delta_b)
