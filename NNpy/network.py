import numpy as np
from layer import Layer
from activation_functions import ActivationFunction


class NeuralNetwork:

    def __init__(self, layer_sizes: list, input_size: int, act_h: ActivationFunction, act_o: ActivationFunction,
                 w_init: str,
                 ETA: float, LAMBDA: float, ALPHA: float, loss: str, task_type: str):
        self.task_type = task_type
        self.loss = loss
        self.ALPHA = ALPHA
        self.ETA = ETA
        self.LAMBDA = LAMBDA

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
            layer.update_w(self.ETA, self.LAMBDA, self.ALPHA)


    def train(self):
        pass
