import numpy as np


class SGD:
    def __init__(self, regularizer, ALPHA, nesterov: bool = False):
        self.nesterov = nesterov
        self.ALPHA = ALPHA
        self.regularizer = regularizer
        pass

    def __call__(self, layer, ETA):
        if self.nesterov:
            # apply the momentum
            nest_w = layer.w + self.ALPHA * layer.delta_w
            # new delta
            layer.delta = layer.back * layer.act_fun.derivative(np.dot(nest_w, np.append(layer.x, 1)))
        layer.delta_w = ETA * np.dot(np.transpose(layer.x), layer.delta) + self.ALPHA * layer.delta_w
        # todo: remove bias b from regularizer or provide a different lambda for it
        layer.w = layer.w + layer.delta_w + self.regularizer(layer.w) #- LAMBDA * layer.w
