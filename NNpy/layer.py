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

    """

    def __init__(self, num_unit: int, num_input, act_fun: ActivationFunction, w_init:str):
        self.x = np.zeros((num_input, 1))
        self.net = np.zeros((num_input, 1))
        self.delta_w = np.zeros((num_unit, num_input+1))
        self.w = winit.weights_init(w_init, num_unit=num_unit, num_input=num_input)
        self.act_fun = act_fun

    def feed_forward(self, x):
        """
        Computes the output of the layer

        Stores the input x and the net result

        :param x: input matrix
        :return: output of the layer
        """
        self.x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        self.net = np.dot(self.x, np.transpose(self.w))  # add 1 as bias input
        return self.act_fun.output(self.net)

    def back_propagate(self, back):

        """
        Computes δ = back * f'(net)

        :param back: back propagated sum
        :return: δ w to back propagate to previous layer
        """
        self.back = back
        self.delta = back * self.act_fun.derivative(self.net)


        # send to prev layer
        return np.dot(self.delta, self.w[0:, :-1])



    '''def update_w(self, ETA, LAMBDA, ALPHA, nesterov=False):
        """
        To update weights we use x=O_u in (Δ w_t,u = δ_t * O_u)
        :param nesterov:
        :param ETA: learning rate
        :param LAMBDA: lambda regulariztion
        :param ALPHA: momentum
        :return:

        momentum =  Δ w_new = η δ x + α Δ w_old
                    w_new = w + Δ w_new - λ w

        nesterov =  We calculate the gradient not with respect to the current step but with respect to the future step.
                    1) apply the momentum w_ = w + α Δ w_old
                    2) evaluate the gradient using w_
                    3) compute and apply Δ w summing momentum and new gradient

        bias w0 is omitted from the regularizer (because
        its inclusion  causes the results to be not independent from target
        shift/scaling) or it may be included but with its own regularization
        coefficient (see Bishop book, Hastie et al. book)

        divide by l only the gradient of E (i.e. 𝜂) where you use a sum over the patterns

        """
        if nesterov:
            # apply the momentum
            nest_w = self.w + ALPHA * self.delta_w
            # new delta
            self.delta = self.back * self.act_fun.derivative(np.dot(nest_w, np.append(self.x, 1)))
        self.delta_w = ETA * np.dot(np.transpose(self.x), self.delta) + ALPHA * self.delta_w
        # todo: remove bias from regularizer
        self.w = self.w + self.delta_w - LAMBDA * self.w'''
