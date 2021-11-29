import numpy as np
from layer import Layer
from activation_functions import ActivationFunction


class NeuralNetwork:

    def __init__(self, layer_sizes: list, input_size: int, act_h: ActivationFunction, act_o: ActivationFunction,
                 w_init: str,
                 ETA: float, LAMBDA: float, ALPHA: float, loss: str, task_type: str, optimizer):
        """
        :param layer_sizes:
        :param input_size:
        :param act_h:
        :param act_o:
        :param w_init:
        :param ETA:
        :param LAMBDA:
        :param ALPHA:
        :param loss:
        :param task_type:   regression -> output linear units
                            classification -> 1-of-k multioutput
                                              Sigmoid → Often symmetric logistic (TanH) learn faster 0.9 can be used instead of  1
                                                (as d value) to avoid asymptotical convergence
                                                (-0.9 for –1 for TanH, or 0.1 for 0 for logistic)
                                             FOR 0/1 softmax
        """
        self.task_type = task_type
        self.loss = loss
        #self.ALPHA = ALPHA
        self.ETA = ETA
        #self.LAMBDA = LAMBDA
        self.optimizer = optimizer

        # initialize layers
        self.layers = []
        for i, l in enumerate(reversed(layer_sizes.append(input_size)[:-1])):
            self.layers.append(Layer(l, layer_sizes[i + 1], act_o if i == 0 else act_h, w_init))
        self.layers.reverse()

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x

    def back_propagate(self, back):
        for layer in reversed(self.layers):
            back = layer.back_propagate(back)
            layer.update_w(self.ETA, self.optimizer)
            #layer.update_w(self.ETA, self.LAMBDA, self.ALPHA)

    def train(self, data_in, data_out):
        """
        for mini batch consider the gradient over different examples  (moving average of the past gradients)

        :param data_in:
        :param data_out:
        :return:
        """
        nn_out = self.feed_forward(data_in)
        diff = data_out - nn_out
        self.back_propagate(diff)
