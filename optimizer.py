import numpy as np


class SGDOptimizer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def update(self, gradients_W, gradients_b):
        for i, layer in enumerate(self.model.layers):
            layer.weights -= self.learning_rate * gradients_W[i]
            layer.biases -= self.learning_rate * gradients_b[i]
    
    def __repr__(self):
        return f"SGD Optimizer - Learning Rate: {self.learning_rate}"


class MomentumGDOptimizer:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocities_W = [np.zeros_like(layer.weights) for layer in model.layers]
        self.velocities_b = [np.zeros_like(layer.biases) for layer in model.layers]

    def update(self, gradients_W, gradients_b):
        for i, layer in enumerate(self.model.layers):
            self.velocities_W[i] = (self.momentum * self.velocities_W[i]) + ((1 - self.momentum) * gradients_W[i])
            self.velocities_b[i] = (self.momentum * self.velocities_b[i]) + ((1 - self.momentum) * gradients_b[i])

            layer.weights -= (self.learning_rate * self.velocities_W[i])
            layer.biases -= (self.learning_rate * self.velocities_b[i])
    
    def __repr__(self):
        return f"MomentumGD Optimizer - Learning Rate: {self.learning_rate} - Momentum: {self.momentum}"


class NAGOptimizer:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocities_W = [np.zeros_like(layer.weights) for layer in model.layers]
        self.velocities_b = [np.zeros_like(layer.biases) for layer in model.layers]

    def update(self, gradients_W, gradients_b):
        for i, layer in enumerate(self.model.layers):
            lookahead_W = layer.weights - self.momentum * self.velocities_W[i]
            grad_W = gradients_W[i] + self.weight_decay * lookahead_W

            self.velocities_W[i] = (self.momentum * self.velocities_W[i]) + (self.learning_rate * grad_W)
            self.velocities_b[i] = (self.momentum * self.velocities_b[i]) + (self.learning_rate * gradients_b[i])

            layer.weights -= self.velocities_W[i]
            layer.biases -= self.velocities_b[i]
    
    def __repr__(self):
        return f"NAG Optimizer - Learning Rate: {self.learning_rate} - Momentum: {self.momentum}"


class RMSPropOptimizer:
    def __init__(self, model, learning_rate=0.001, beta1=0.9, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta = beta1
        self.epsilon = epsilon

        self.sq_grads_W = [np.zeros_like(layer.weights) for layer in model.layers]
        self.sq_grads_b = [np.zeros_like(layer.biases) for layer in model.layers]

    def update(self, gradients_W, gradients_b):
        for i, layer in enumerate(self.model.layers):
            self.sq_grads_W[i] = self.beta * self.sq_grads_W[i] + (1 - self.beta) * (gradients_W[i] ** 2)
            self.sq_grads_b[i] = self.beta * self.sq_grads_b[i] + (1 - self.beta) * (gradients_b[i] ** 2)

            layer.weights -= self.learning_rate * gradients_W[i] / (np.sqrt(self.sq_grads_W[i]) + self.epsilon)
            layer.biases -= self.learning_rate * gradients_b[i] / (np.sqrt(self.sq_grads_b[i]) + self.epsilon)

    def __repr__(self):
        return f"RMSProp Optimizer - Learning Rate: {self.learning_rate} - Beta: {self.beta} - Epsilon: {self.epsilon}"


class AdamOptimizer:
    def __init__(self, model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_W = [np.zeros_like(layer.weights) for layer in model.layers]
        self.m_b = [np.zeros_like(layer.biases) for layer in model.layers]
        self.v_W = [np.zeros_like(layer.weights) for layer in model.layers]
        self.v_b = [np.zeros_like(layer.biases) for layer in model.layers]
        self.t = 0

    def update(self, gradients_W, gradients_b):
        self.t += 1
        for i, layer in enumerate(self.model.layers):
            self.m_W[i] = (self.beta1 * self.m_W[i]) + ((1 - self.beta1) * gradients_W[i])
            self.m_b[i] = (self.beta1 * self.m_b[i]) + ((1 - self.beta1) * gradients_b[i])

            self.v_W[i] = (self.beta2 * self.v_W[i]) + ((1 - self.beta2) * (gradients_W[i] ** 2))
            self.v_b[i] = (self.beta2 * self.v_b[i]) + ((1 - self.beta2) * (gradients_b[i] ** 2))

            m_W_hat = self.m_W[i] / (1 - (self.beta1 ** self.t))
            m_b_hat = self.m_b[i] / (1 - (self.beta1 ** self.t))
            v_W_hat = self.v_W[i] / (1 - (self.beta2 ** self.t))
            v_b_hat = self.v_b[i] / (1 - (self.beta2 ** self.t))

            layer.weights -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            layer.biases -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
    
    def __repr__(self):
        return f"Adam Optimizer - Learning Rate: {self.learning_rate} - Beta1: {self.beta1} - Beta2: {self.beta2} - Epsilon: {self.epsilon}"


class NadamOptimizer:
    def __init__(self, model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_W = [np.zeros_like(layer.weights) for layer in model.layers]
        self.m_b = [np.zeros_like(layer.biases) for layer in model.layers]
        self.v_W = [np.zeros_like(layer.weights) for layer in model.layers]
        self.v_b = [np.zeros_like(layer.biases) for layer in model.layers]
        self.t = 0

    def update(self, gradients_W, gradients_b):
        self.t += 1
        for i, layer in enumerate(self.model.layers):
            self.m_W[i] = (self.beta1 * self.m_W[i]) + ((1 - self.beta1) * gradients_W[i])
            self.m_b[i] = (self.beta1 * self.m_b[i]) + ((1 - self.beta1) * gradients_b[i])

            self.v_W[i] = (self.beta2 * self.v_W[i]) + ((1 - self.beta2) * (gradients_W[i] ** 2))
            self.v_b[i] = (self.beta2 * self.v_b[i]) + ((1 - self.beta2) * (gradients_b[i] ** 2))

            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            m_W_nadam = self.beta1 * m_W_hat + (1 - self.beta1) * gradients_W[i] / (1 - self.beta1 ** self.t)
            m_b_nadam = self.beta1 * m_b_hat + (1 - self.beta1) * gradients_b[i] / (1 - self.beta1 ** self.t)

            layer.weights -= self.learning_rate * m_W_nadam / (np.sqrt(v_W_hat) + self.epsilon)
            layer.biases -= self.learning_rate * m_b_nadam / (np.sqrt(v_b_hat) + self.epsilon)
    
    def __repr__(self):
        return f"Nadam Optimizer - Learning Rate: {self.learning_rate} - Beta1: {self.beta1} - Beta2: {self.beta2} - Epsilon: {self.epsilon}"

class OptimizerFactory:
    OPTIMIZERS = {
        "sgd": (SGDOptimizer, ["learning_rate", "weight_decay"]),
        "momentum": (MomentumGDOptimizer, ["learning_rate", "momentum", "weight_decay"]),
        "nesterov": (NAGOptimizer, ["learning_rate", "momentum", "weight_decay"]),
        "rmsprop": (RMSPropOptimizer, ["learning_rate", "beta", "epsilon", "weight_decay"]),
        "adam": (AdamOptimizer, ["learning_rate", "beta1", "beta2", "epsilon", "weight_decay"]),
        "nadam": (NadamOptimizer, ["learning_rate", "beta1", "beta2", "epsilon", "weight_decay"]),
    }

    @staticmethod
    def get_optimizer(name, model, **kwargs):
        name = name.lower()
        if name not in OptimizerFactory.OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {name}. Available options: {list(OptimizerFactory.OPTIMIZERS.keys())}")

        optimizer_class, valid_params = OptimizerFactory.OPTIMIZERS[name]

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        return optimizer_class(model, **filtered_kwargs)