import numpy as np


class SGD:
    def __init__(self, learning_rate, nesterov: bool = False):
        pass

    def update_w(self, layer, ETA, regularizer, ALPHA=0, nesterov=False):
        if nesterov:
            # apply the momentum
            nest_w = layer.w + ALPHA * layer.delta_w
            # new delta
            layer.delta = layer.back * layer.act_fun.derivative(np.dot(nest_w, np.append(layer.x, 1)))
        layer.delta_w = ETA * np.dot(np.transpose(layer.x), layer.delta) + ALPHA * layer.delta_w
        # todo: remove bias from regularizer
        layer.w = layer.w + layer.delta_w - LAMBDA * layer.w
