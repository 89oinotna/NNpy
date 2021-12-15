import numpy as np
from layer import Layer
import activation_functions as af
import losses as l
import metrics as m
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
            - epochs (int): number of epochs  (Default None) if none means that we are using another way to stop
    """

    def __init__(self, layer_sizes: list, input_size: int, act_hidden: af.ActivationFunction,
                 act_out: af.ActivationFunction,
                 w_init: str, loss: l.Loss, metric: m.Metric,
                 optimizer: opt.Optimizer = opt.SGD(weight_regularizer=reg.Tikonov(0.5)), epochs: int = None):

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

    @staticmethod
    def init(layer_sizes: list, input_size: int, act_hidden: str, act_out: str,
             w_init: str, loss: str, metric: str, optimizer: str, epochs=None, **kwargs):
        """
        Initializes a fully connected feed forward neural network implementation

        :param layer_sizes: list of sizes for the layers, ordered first to last
        :param input_size: size of the input
        :param act_hidden: activation function used in hidden layers (see activation_functions.py)
        :param act_out: activation function used in the output layer (see activation_functions.py)
        :param w_init: weight initialization method (see weights_init) (see weights_init)
        :param loss: loss used for the training (see losses)
        :param metric: metric used for evaluation (see metrics)
        :param optimizer: optimizer used to learn (Note that you will need also to insert
                                    parameters for the optimizer, see its implementation) (see optimizers)
        :param epochs: number of epochs  (Default None) if none means that we are using another way to stop
        :param kwargs: other arguments needed for instance by the optimizer
        :return:
        """
        act_hidden = af.activation_function(act_hidden, **kwargs)
        act_out = af.activation_function(act_out, **kwargs)
        loss = l.loss(loss, **kwargs)
        metric = m.metric(metric, **kwargs)
        optimizer = opt.optimizer(optimizer, **kwargs)
        return NeuralNetwork(layer_sizes, input_size, act_hidden, act_out, w_init, loss, metric, optimizer, epochs)

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

        :param tr_data: training data
        :param tr_label: training labels
        :param vl_data: validation data
        :param vl_label: validation labels
        :return: (tr_metric: list, tr_loss: list), (vl_metric: list, vl_loss: list) if a vl set has been passed
                tr_metric: list, tr_loss: list otherwise
        """
        tr_loss = []
        tr_metric = []
        vl_loss = []
        vl_metric = []

        # todo: for mini batch consider the gradient over different examples (moving average of the past gradients)
        for i in range(self.epochs):
            output = self.feed_forward(tr_data)
            diff = self.loss.partial_derivative(tr_label, output)
            self.back_propagate(diff)
            tr_loss.append(self.loss.error(tr_label, output))
            tr_metric.append(self.metric(tr_label, output))

            if vl_data is not None:
                output = self.feed_forward(tr_data)
                vl_loss.append(self.loss.error(vl_label, output))
                vl_metric.append(self.metric(vl_label, output))

        if vl_data is not None:
            return (tr_metric, tr_loss), (vl_metric, vl_loss)
        return tr_metric, tr_loss
