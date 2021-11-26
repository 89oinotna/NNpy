import numpy as np
from activation_functions import ActivationFunction

import weights_init as winit


class Layer:
    """
    This is a test class for dataclasses.

    This is the body of the docstring description.

        Args:
            - num_unit (int): number of units of the layer.
            - num_input (int): size of the input.
            - act_fun: activation function used by units of the layer
            - w_init: weight initialization method

    """

    def __init__(self, num_unit: int, num_input, act_fun: ActivationFunction, w_init):
        self.x = np.zeros((num_input, 0))
        self.net = np.zeros((num_input, 0))
        self.w = winit.weights_init(w_init, num_unit=num_unit, num_input=num_input)
        self.act_fun = act_fun

    def feed_forward(self, x):
        """

        :param x: input
        :return:
        """
        self.x = x
        self.net = np.dot(self.w, np.append(x, 1))
        self.act_fun.output(self.net)
        return self.act_fun.output(self.net)

    def back_propagate(self, back):
        self.delta = back * self.act_fun.derivative(self.net)
        return np.dot(self.delta, np.transpose(self.w))

    def update_w(self, ETA, LAMBDA, ALPHA, momentum):
        """
        To update weights we use x=O_u in (Δ w_t,u = δ_t * O_u)
        :param ETA: learning rate
        :param LAMBDA: lambda regulariztion
        :param ALPHA: momentum
        :return:

        momentum = Δ w_new = η δ x + α Δ w_old

        nesterov =  1) apply the momentum w_ = w + α Δ w_old
                    2) evaluate the gradient using w_
                    3) compute and apply Δ w summing momentum and new gradient

        """
        delta_w = np.dot(np.transpose(self.x), self.delta)
