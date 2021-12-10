import numpy as np
from regularization import WeightRegularizer


class Optimizer:
    def __call__(self, *args, **kwargs):
        pass


class SGD(Optimizer):
    # todo implement variable learning rate
    def __init__(self, ETA = 0.01, weight_regularizer: WeightRegularizer = None, ALPHA: float = 0.5, nesterov: bool = False):
        self.ETA = ETA
        self.nesterov = nesterov
        self.ALPHA = ALPHA
        self.weight_regularizer = weight_regularizer
        pass

    def __call__(self, layer):
        """
            Loss = error + penalty to separate eta
        """
        if self.nesterov:
            # apply the momentum
            nest_w = layer.w + self.ALPHA * layer.delta_w
            # new delta
            layer.delta = layer.back * layer.act_fun.partial_derivative(np.dot(nest_w, layer.x))

        #print(np.dot(np.transpose(layer.x), layer.delta))
        layer.delta_w = self.ETA * np.dot(np.transpose(layer.delta), layer.x)
        layer.delta_w += self.ALPHA * layer.delta_w # momentum
        layer.w += layer.delta_w

        # remove bias b from regularizer or provide a different lambda for it
        if self.weight_regularizer is not None:
            layer.w[0:, :-1] = self.weight_regularizer(layer.w[0:, :-1])  # - LAMBDA * layer.w

        layer.w *= (1/layer.delta.shape[0])
