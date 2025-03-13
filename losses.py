import numpy as np

class CrossEntropyLoss:
    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-8))
    
    def derivative(self, y_true, y_pred):
        return y_pred - y_true

class MeanSquaredErrorLoss:
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class LossFnFactory:
    LOSSES = {
        'cross_entropy': CrossEntropyLoss,
        'mean_squared_error': MeanSquaredErrorLoss
    }

    @staticmethod
    def get(loss):
        loss = loss.lower()
        if loss not in LossFnFactory.LOSSES:
            print(f"Loss function {loss} not supported. Defaulting to CrossEntropyLoss.")
            loss = 'cross_entropy'
        return LossFnFactory.LOSSES[loss]()