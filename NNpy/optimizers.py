import numpy as np
from regularization import WeightRegularizer


class SGD:
    # todo implement variable learning rate
    def __init__(self, ETA, weight_regularizer: WeightRegularizer = None, ALPHA: float = 0, nesterov: bool = False):
        self.ETA = ETA
        self.nesterov = nesterov
        self.ALPHA = ALPHA
        self.weight_regularizer = weight_regularizer
        pass

    def __call__(self, layer):
        if self.nesterov:
            # apply the momentum
            nest_w = layer.w + self.ALPHA * layer.delta_w
            # new delta
            layer.delta = layer.back * layer.act_fun.derivative(np.dot(nest_w, np.append(layer.x, 1)))

        layer.delta_w = self.ETA * np.dot(np.transpose(layer.x), layer.delta) + self.ALPHA * layer.delta_w
        # todo: remove bias b from regularizer or provide a different lambda for it
        layer.w = layer.w + layer.delta_w
        layer.w = self.weight_regularizer(layer.w)  # - LAMBDA * layer.w
