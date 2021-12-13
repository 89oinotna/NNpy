import numpy as np
import regularization as reg


def optimizer(type_init, **kwargs):
    init = {
        'sgd': SGD
    }
    if 'weight_regularizer' in kwargs:
        kwargs['weight_regularizer'] = reg.regularizer(kwargs['weight_regularizer'], **kwargs)
    matrix = init[type_init](**kwargs)
    return matrix


class Optimizer:
    def __call__(self, *args, **kwargs):
        pass


class SGD(Optimizer):
    # todo implement variable learning rate
    def __init__(self, ETA = 0.01, weight_regularizer: reg.WeightRegularizer = None, ALPHA: float = 0.5, nesterov: bool = False, **kwargs):
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
            layer.delta = layer.back * layer.act_fun.derivative(np.dot(layer.x, np.transpose(nest_w)))

        # no momentum on bias
        layer.delta_w[0:, :-1] = self.ALPHA * layer.delta_w[0:, :-1] # uses old delta_w so must be the first to update it
        layer.delta_w += self.ETA * (1/layer.delta.shape[0]) * np.dot(np.transpose(layer.delta), layer.x)

        # remove bias b from regularizer or provide a different lambda for it
        if self.weight_regularizer is not None:
            layer.w[0:, :-1] = self.weight_regularizer(layer.w[0:, :-1])   # - LAMBDA * layer.w

        layer.w += layer.delta_w


