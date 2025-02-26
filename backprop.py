import numpy as np

def cross_entropy_loss(y_hats, class_label):
    return -np.log(y_hats[class_label])

# gradient wrt to outer layer
def gradient_wrt_output(y_hats, class_label):
    y_hats[class_label] -= 1
    return y_hats

# gradient wrt to hidden layer
def gradient_wrt_h(w_next, gradient_a_next):
    return np.dot(w_next.T, gradient_a_next)

# gradient wrt to pre-activation
def gradient_wrt_a(gradient_h, a):
    def sigmoid_prime(a):
        return a * (1 - a)
    
    return gradient_h * sigmoid_prime(a)

# gradient wrt to weights
def gradient_wrt_w(gradient_a, h):
    return np.outer(gradient_a, h)

# gradient wrt to bias
def gradient_wrt_b(gradient_a):
    return gradient_a