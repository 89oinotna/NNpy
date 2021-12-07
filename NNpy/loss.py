import numpy as np


class Loss:
    def error(self, label, output):
        pass

    def partial_derivative(self, label, output):
        pass


class MeanSquaredError(Loss):
    def error(self, label, output):
        np.mean(np.square(label - output))

    def partial_derivative(self, label, output):
        return output - label
