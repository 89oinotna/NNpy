import numpy as np
import regularization as reg


def optimizer(type_init, **kwargs):
    init = {
        'sgd': SGD
    }
    if 'weight_regularizer' in kwargs:
        kwargs['weight_regularizer'] = reg.regularizer(**kwargs['weight_regularizer'])
    matrix = init[type_init](**kwargs)
    return matrix


class Optimizer:
    def __call__(self, *args, **kwargs):
        pass


class SGD(Optimizer):
    """
    todo implement variable learning rate
    Using mini-batch → the gradient does not decrease to zero close to a
minimum (as the exact gradient can do)
• Hence fixed learning rate should be avoided
    """

    def __init__(self, ETA=0.01, weight_regularizer: reg.WeightRegularizer = None, ALPHA: float = 0.5,
                 nesterov: bool = False, momentum_window: int = None):
        self.ETA = ETA
        self.nesterov = nesterov
        self.ALPHA = ALPHA
        self.weight_regularizer = weight_regularizer
        self.momentum_window = None
        if momentum_window is not None:
            self.momentum_window = momentum_window  # used for minibatch (moving average)

    def __call__(self, layer):
        """
            Loss = error + penalty to separate eta
        """
        if self.nesterov:
            # apply the momentum
            nest_w = layer.w + self.ALPHA * layer.old_delta_w
            # new delta
            layer.delta = layer.back * layer.act_fun.derivative(np.dot(layer.x, np.transpose(nest_w)))

        # no momentum on bias
        layer.delta_w[0:, :-1] = self.ALPHA * layer.old_delta_w[0:, :-1]

        layer.delta_w += self.ETA * (1 / layer.delta.shape[0]) * np.dot(np.transpose(layer.delta), layer.x)

        # remove bias b from regularizer or provide a different lambda for it
        if self.weight_regularizer is not None:
            layer.w[0:, :-1] = self.weight_regularizer(layer.w[0:, :-1])  # - LAMBDA * layer.w

        layer.w += layer.delta_w

        # momentum for mini batch should consider the gradient over different examples
        # (moving average of the past gradients)
        if self.momentum_window is not None:
            layer.old_delta_w = (layer.old_delta_w * len(layer.list_old_delta_w)) + layer.delta_w
            # only if the list has all the values
            if not (len(layer.list_old_delta_w) < self.momentum_window):
                layer.old_delta_w -= layer.list_old_delta_w[0]
                layer.list_old_delta_w=layer.list_old_delta_w[1:]
                layer.list_old_delta_w.append(layer.delta_w)
            else:
                layer.list_old_delta_w.append(layer.delta_w)
            layer.old_delta_w /= len(layer.list_old_delta_w)

        else:
            layer.old_delta_w = layer.delta_w
