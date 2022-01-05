import numpy as np
from activation_functions import ActivationFunction

import weights_init as winit


class Layer:
    """
    Class that identifies the layer of a fully connected neural network

    Each layer has its weights matrix w

        Args:
            - num_unit (int): number of units of the layer.
            - num_input (int): size of the input.
            - act_fun: activation function used by units of the layer
            - w_init: weight initialization method
            - minibatch

    """

    def __init__(self, num_unit: int, num_input, act_fun: ActivationFunction, w_init:str, minibatch: bool = False):
        self.delta_w = np.zeros((num_unit, num_input+1)) # + bias
        self.old_delta_w = np.zeros((num_unit, num_input+1)) # + bias
        if minibatch:
            self.list_old_delta_w = [self.old_delta_w]
        self.w = winit.weights_init(w_init, num_unit=num_unit, num_input=num_input)
        self.act_fun = act_fun

    def feed_forward(self, x):
        """
        Computes the output of the layer

        Stores the input x and the net result

        :param x: input matrix
        :return: output of the layer
        """
        self.x = np.append(x, np.ones((x.shape[0], 1)), axis=1) # add 1 as bias input
        self.net = np.dot(self.x, np.transpose(self.w))
        return self.act_fun.output(self.net)

    def back_propagate(self, back):

        """
        Computes δ = back * f'(net)

        :param back: back propagated sum
        :return: δ w to back propagate to previous layer
        """
        self.back=back

        self.delta = back * self.act_fun.derivative(self.net)

        # send to prev layer
        return np.dot(self.delta, self.w[:, :-1])


