import numpy as np

class SGDOptimizer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def update(self, gradients_W, gradients_b):
        if len(gradients_W) != len(self.model.weights):
            raise ValueError("Invalid number of gradients provided for weights")
        if len(gradients_b) != len(self.model.biases):
            raise ValueError("Invalid number of gradients provided for biases")
        
        for i in range(len(self.model.weights)):
            self.model.weights[i] -= self.learning_rate * gradients_W[i]
            self.model.biases[i] -= self.learning_rate * gradients_b[i]

    def __repr__(self):
        return f"SGD Optimizer - Learning Rate: {self.learning_rate}"


class MomentumGDOptimizer:
    def __init__(self, model, learning_rate, momentum):
        self.model = model
        self.momentum = momentum
        self.learning_rate = learning_rate

        self.velocities_W = [np.zeros_like(w) for w in model.weights]
        self.velocities_b = [np.zeros_like(b) for b in model.biases]

    def update(self, gradients_W, gradients_b):
        for i in range(len(self.model.weights)):
            self.velocities_W[i] = (self.momentum * self.velocities_W[i]) + ((1 - self.momentum) * gradients_W[i])
            self.velocities_b[i] = (self.momentum * self.velocities_b[i]) + ((1 - self.momentum) * gradients_b[i])

            self.model.weights[i] -= (self.learning_rate * self.velocities_W[i])
            self.model.biases[i] -= (self.learning_rate * self.velocities_b[i])

    def __repr__(self):
        return f"MomentumGD Optimizer - Learning Rate: {self.learning_rate} - Momentum: {self.momentum}"


class NAGOptimizer:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocities_W = [np.zeros_like(w) for w in model.weights]
        self.velocities_b = [np.zeros_like(b) for b in model.biases]

    def update(self, gradients_W, gradients_b):
        # Lookahead Step: Temporarily move weights in the direction of momentum
        for i in range(len(self.model.weights)):
            self.model.weights[i] += self.momentum * self.velocities_W[i]
            self.model.biases[i] += self.momentum * self.velocities_b[i]

        # Compute gradients at lookahead position
        gradients_W, gradients_b = self.model.backward(self.model.activations[-1])

        # Undo the lookahead step and update weights
        for i in range(len(self.model.weights)):
            self.velocities_W[i] = self.momentum * self.velocities_W[i] + self.learning_rate * gradients_W[i]
            self.velocities_b[i] = self.momentum * self.velocities_b[i] + self.learning_rate * gradients_b[i]

            self.model.weights[i] -= self.velocities_W[i]
            self.model.biases[i] -= self.velocities_b[i]

    def __repr__(self):
        return f"NAG Optimizer - Learning Rate: {self.learning_rate} - Momentum: {self.momentum}"


class RMSPropOptimizer:
    def __init__(self, model, learning_rate, beta1, epsilon):
        self.model = model
        self.learning_rate = learning_rate
        self.beta = beta1
        self.epsilon = epsilon
        self.sq_grads_W = [np.zeros_like(w) for w in model.weights]
        self.sq_grads_b = [np.zeros_like(b) for b in model.biases]

    def update(self, gradients_W, gradients_b):
        for i in range(len(self.model.weights)):
            self.sq_grads_W[i] = (self.beta * self.sq_grads_W[i]) + (1 - self.beta) * (gradients_W[i] ** 2)
            self.sq_grads_b[i] = (self.beta * self.sq_grads_b[i]) + (1 - self.beta) * (gradients_b[i] ** 2)

            self.model.weights[i] -= self.learning_rate * gradients_W[i] / (np.sqrt(self.sq_grads_W[i]) + self.epsilon)
            self.model.biases[i] -= self.learning_rate * gradients_b[i] / (np.sqrt(self.sq_grads_b[i]) + self.epsilon)

    def __repr__(self):
        return f"RMSProp Optimizer - Learning Rate: {self.learning_rate} - Beta: {self.beta} - Epsilon: {self.epsilon}"


class AdamOptimizer:
    def __init__(self, model, learning_rate, beta1, beta2, epsilon):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
        self.m_W = [np.zeros_like(w) for w in model.weights]
        self.m_b = [np.zeros_like(b) for b in model.biases]
        self.v_W = [np.zeros_like(w) for w in model.weights]
        self.v_b = [np.zeros_like(b) for b in model.biases]

    def _update_moments(self, i, grad_W, grad_b):
        self.m_W[i] = (self.beta1 * self.m_W[i]) + ((1 - self.beta1) * grad_W)
        self.m_b[i] = (self.beta1 * self.m_b[i]) + ((1 - self.beta1) * grad_b)
        self.v_W[i] = (self.beta2 * self.v_W[i]) + ((1 - self.beta2) * (grad_W ** 2))
        self.v_b[i] = (self.beta2 * self.v_b[i]) + ((1 - self.beta2) * (grad_b ** 2))

    def _compute_hat_values(self, i):
        m_W_hat = self.m_W[i] / (1 - (self.beta1 ** self.t))
        m_b_hat = self.m_b[i] / (1 - (self.beta1 ** self.t))
        v_W_hat = self.v_W[i] / (1 - (self.beta2 ** self.t))
        v_b_hat = self.v_b[i] / (1 - (self.beta2 ** self.t))
        return m_W_hat, m_b_hat, v_W_hat, v_b_hat
    
    def _update_weights_biases(self, i, m_W_hat, m_b_hat, v_W_hat, v_b_hat):
        self.model.weights[i] -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        self.model.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def update(self, gradients_W, gradients_b):
        self.t += 1
        for i in range(len(self.model.weights)):
            self._update_moments(i, gradients_W[i], gradients_b[i])
            m_W_hat, m_b_hat, v_W_hat, v_b_hat = self._compute_hat_values(i)
            self._update_weights_biases(i, m_W_hat, m_b_hat, v_W_hat, v_b_hat)

    def __repr__(self):
        return f"Adam Optimizer - Learning Rate: {self.learning_rate} - Beta1: {self.beta1} - Beta2: {self.beta2} - Epsilon: {self.epsilon}"


class NadamOptimizer(AdamOptimizer):
    def __init__(self, model, learning_rate, beta1, beta2, epsilon):
        super().__init__(model, learning_rate, beta1, beta2, epsilon)

    def _update_weights_biases(self, i, m_W_hat, m_b_hat, v_W_hat, v_b_hat, grad_W, grad_b):
        self.model.weights[i] -= (self.learning_rate / (np.sqrt(v_W_hat + self.epsilon))) * (((self.beta1 * m_W_hat) + (((1 - self.beta1) * grad_W) / (1 - (self.beta1 ** self.t)))))
        self.model.biases[i] -= (self.learning_rate / (np.sqrt(v_b_hat + self.epsilon))) * (((self.beta1 * m_b_hat) + (((1 - self.beta1) * grad_b) / (1 - (self.beta1 ** self.t)))))

    def update(self, gradients_W, gradients_b):
        self.t += 1
        for i in range(len(self.model.weights)):
            self._update_moments(i, gradients_W[i], gradients_b[i])
            m_W_hat, m_b_hat, v_W_hat, v_b_hat = self._compute_hat_values(i)
            self._update_weights_biases(i, m_W_hat, m_b_hat, v_W_hat, v_b_hat, gradients_W[i], gradients_b[i])

    def __repr__(self):
        return f"Nadam Optimizer - Learning Rate: {self.learning_rate} - Beta1: {self.beta1} - Beta2: {self.beta2} - Epsilon: {self.epsilon}"


class OptimizerFactory:
    OPTIMIZERS = {
        "sgd": (SGDOptimizer, ["learning_rate", "weight_decay"]),
        "momentum": (MomentumGDOptimizer, ["learning_rate", "momentum", "weight_decay"]),
        "nesterov": (NAGOptimizer, ["learning_rate", "momentum", "weight_decay"]),
        "rmsprop": (RMSPropOptimizer, ["learning_rate", "beta1", "epsilon", "weight_decay"]),
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