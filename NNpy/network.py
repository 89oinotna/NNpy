import numpy as np
from layer import Layer
from activation_functions import ActivationFunction
from loss import Loss
from metrics import Metric


class NeuralNetwork:
    """
    task_type:  regression -> output linear units
                classification -> 1-of-k multioutput
                                  Sigmoid → Often symmetric logistic (TanH) learn faster 0.9 can be used instead of  1
                                    (as d value) to avoid asymptotical convergence
                                    (-0.9 for –1 for TanH, or 0.1 for 0 for logistic)
                                 FOR 0/1 softmax
    """

    def __init__(self, layer_sizes: list, input_size: int, act_hidden: ActivationFunction, act_out: ActivationFunction,
                 w_init: str, loss: Loss, metric: Metric, optimizer):
        """
        :param layer_sizes:
        :param input_size:
        :param act_hidden:
        :param act_out:
        :param w_init:
        :param loss:
        """

        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        # initialize layers
        self.layers = []
        for i, l in enumerate(reversed(layer_sizes.append(input_size)[:-1])):
            self.layers.append(Layer(l, layer_sizes[i + 1], act_out if i == 0 else act_hidden, w_init))
        self.layers.reverse()

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x

    def back_propagate(self, back):
        for layer in reversed(self.layers):
            back = layer.back_propagate(back)
            self.optimizer(layer)

    def train(self, data_in, label):
        """
        for mini batch consider the gradient over different examples
        (moving average of the past gradients)

        :param data_in:
        :param label:
        :return:
        """
        output = self.feed_forward(data_in)
        diff = self.loss.derivative(label, output)
        self.back_propagate(diff)
