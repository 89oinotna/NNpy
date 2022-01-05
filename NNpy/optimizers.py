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
    Class that identifies the SGD optimizer

        Args:
            - ETA (float): learning rate
            - weight_regularizer (WeightRegularizer): weight regularizer (optional)
            - ALPHA (float): for the momentum (eg 0 < 0.5 - 0.9 < 1) allows to use higher eta
            - nesterov (bool): True for nesterov momentum
            - momentum_window (int): size of the window to consider for the momentum when using minibatch
            - variable_eta (dict): when using minibatch (should contain: eta_taua and tau)

    """

    def __init__(self, ETA=0.01, weight_regularizer: reg.WeightRegularizer = None, ALPHA: float = 0.0,
                 nesterov: bool = False, momentum_window: int = None, variable_eta: dict = None):
        if variable_eta is not None:
            variable_eta['eta'] = ETA  # eta 0
            variable_eta['step'] = 0
        self.variable_eta = variable_eta
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
                layer.list_old_delta_w = layer.list_old_delta_w[1:]
                layer.list_old_delta_w.append(layer.delta_w)
            else:
                layer.list_old_delta_w.append(layer.delta_w)
            layer.old_delta_w /= len(layer.list_old_delta_w)

        else:
            layer.old_delta_w = layer.delta_w

        if self.variable_eta is not None:
            self.decay_learning_rate()

    def decay_learning_rate(self):
        """
        Using mini-batch → the gradient does not decrease to zero close to a
        minimum (as the exact gradient can do)
        • Hence fixed learning rate should be avoided:
        Set-up: eta_tau as ~1% of eta_0, tau few hundred steps
        """
        if self.variable_eta['step'] < self.variable_eta['tau']:
            self.variable_eta['step'] += 1
            alpha = self.variable_eta['step'] / self.variable_eta['tau']
            self.ETA = (1 - alpha) * self.variable_eta['eta'] + alpha * self.variable_eta['eta_tau']
