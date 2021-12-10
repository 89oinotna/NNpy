import numpy as np
from layer import Layer
from activation_functions import ActivationFunction
from loss import Loss
from metrics import Metric
import optimizers as opt
import regularization as reg

'''
regression: output linear units 
    
    classification: 
        - 1-of-k multioutput use sigmoid, often symmetric logistic (TanH) learn faster 0.9 can be used instead of 1 
        (as d value) to avoid asymptotical convergence (-0.9 for –1 for TanH, or 0.1 for 0 for logistic)  
        
        - for 0/1 softmax 
'''


class NeuralNetwork:
    """
    Fully connected feed forward neural network implementation

          Args:
            - layer_sizes (list): list of sizes for the layers, ordered first to last
            - input_size (int): size of the input
            - act_hidden (ActivationFunction): activation function used in hidden layers
            - act_out (ActivationFunction): activation function used in the output layer
            - w_init (str): weight initialization method (see weights_init)
            - loss (Loss): loss used for the training
            - metric (Metric): metric used for evaluation
            - optimizer (Optimizer): optimizer used to learn
    """

    def __init__(self, layer_sizes: list, input_size: int, act_hidden: ActivationFunction, act_out: ActivationFunction,
                 w_init: str, loss: Loss, metric: Metric,  optimizer: opt.Optimizer = opt.SGD(weight_regularizer=reg.Tikonov(0.5)), epochs=None):

        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.epochs = epochs
        # initialize layers
        self.layers = []
        # reverse and add the input size

        layer_sizes.reverse()
        layer_sizes.append(input_size)
        """ 
        starting from last layer size create each layer using the current value as its unit number 
        and the next value as its input size
        """
        for i, l in enumerate(layer_sizes[:-1]):
            self.layers.append(Layer(l, layer_sizes[i + 1], act_out if i == 0 else act_hidden, w_init))
        # reverse the layers since they are created backward and remove the input size
        self.layers.reverse()

    def feed_forward(self, x):
        """
        feed forward implementation through the layers
        :param x: input of the network
        :return: output of the network
        """
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x

    def back_propagate(self, back):
        """
        back propagation implementation with the given optimizer
        :param back: loss
        """
        for layer in reversed(self.layers):
            back = layer.back_propagate(back)
            self.optimizer(layer)

    def fit(self, tr_data, tr_label, vl_data=None, vl_label=None):
        """
        training algorithm
        :param tr_data:
        :param tr_label:
        :param vl_data:
        :param vl_label:
        :return:
        """
        self.tr_res = []
        self.vl_res = []

        # todo: for mini batch consider the gradient over different examples (moving average of the past gradients)
        for i in range(self.epochs):
            output = self.feed_forward(tr_data)
            diff = self.loss.partial_derivative(tr_label, output)
            self.back_propagate(diff)
            self.tr_res.append(self.metric(tr_label, output))

            if vl_data is not None:
                output = self.feed_forward(tr_data)
                self.vl_res.append(self.metric(vl_label, output))

        return self.tr_res
