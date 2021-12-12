import numpy as np


def loss(type_init, **kwargs):
    init = {
        'mse': lambda **kwargs: MeanSquaredError()
    }
    matrix = init[type_init](**kwargs)
    return matrix


class Loss:
    def error(self, label, output):
        pass

    def partial_derivative(self, label, output):
        pass


class MeanSquaredError(Loss):
    def error(self, label, output):
        return np.mean(np.square(label-output))

    def partial_derivative(self, label, output):
        return label - output
