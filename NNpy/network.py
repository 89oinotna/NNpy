import numpy as np
import pandas as pd
from layer import Layer
import activation_functions as af
import losses
import metrics
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
                 w_init: str, loss: losses.Loss, metric: metrics.Metric,
                 optimizer: opt.Optimizer, epochs: int = None,
                 minibatch_size=None):

        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        # initialize layers
        self.layers = []
        layer_sizes = list(layer_sizes)
        # reverse and add the input size
        layer_sizes.reverse()
        layer_sizes.append(input_size)
        """ 
        starting from last layer size create each layer using the current value as its unit number 
        and the next value as its input size
        """
        for i, l in enumerate(layer_sizes[:-1]):
            self.layers.append(Layer(l, layer_sizes[i + 1], act_out if i == 0 else act_hidden, w_init,
                                     True if minibatch_size is not None else False))
        # reverse the layers since they are created backward and remove the input size
        self.layers.reverse()

    @staticmethod
    def init(layer_sizes: list, input_size: int, act_hidden: str, act_out: str,
             w_init: str, loss: str, metric: str, optimizer: dict, epochs=None,
             minibatch_size: int = None, **kwargs):
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
        :param weight_regularizer: weight regularizer for the optimizer (can be None)
        :param minibatch_size: minibatch size (not needed for batch)
        :param epochs: number of epochs  (Default None) if none means that we are using another way to stop
        :param kwargs: other arguments needed for instance by the optimizer
        :return:


        """
        act_hidden = af.activation_function(act_hidden, **kwargs)
        act_out = af.activation_function(act_out, **kwargs)
        loss = losses.loss(loss, **kwargs)
        metric = metrics.metric(metric, **kwargs)
        optimizer = opt.optimizer(**optimizer)
        return NeuralNetwork(layer_sizes, input_size, act_hidden, act_out, w_init, loss, metric, optimizer, epochs,
                             minibatch_size)

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

    def step(self, data, label):
        output = self.feed_forward(data)
        diff = self.loss.partial_derivative(label, output)
        self.back_propagate(diff)
        return output

    def fit(self, tr_data, tr_label, vl_data=None, vl_label=None, early_stopping=False, monitor="vl_loss",
            min_delta=0.001, patience=20, mode="min"):
        """
        training algorithm

        :param tr_data: training data
        :param tr_label: training labels
        :param vl_data: validation data
        :param vl_label: validation labels
        :param early_stopping: early stopping
        :param monitor: "vl_loss" or "vl_metric"
        :param min_delta: threshold to consider improvements
        :param patience: max number of not improvements after which we stop
        :param mode: "min" or "max" where we select if we want to minimize or maximize
        :return: (tr_metric: list, tr_loss: list), (vl_metric: list, vl_loss: list) if a vl set has been passed
                tr_metric: list, tr_loss: list otherwise
        """

        tr_loss = []
        tr_metric = []
        vl_loss = []
        vl_metric = []
        monitoring = None
        tr = None

        if self.minibatch_size is not None:  # to redo the splitting
            tr = pd.DataFrame(np.concatenate((tr_data, tr_label), axis=1))

        if vl_data is not None and early_stopping:
            if mode == "min":
                best_cost = np.infty
                mode = min
            if mode == "max":
                best_cost = -1
                mode = max
            best_model: dict = {
                "layers": None,
                "loss": None,
                "optimizer": None
            }
            last_improvement = 0
            if monitor == 'vl_loss':
                monitoring = vl_loss
            elif monitor == 'vl_metric':
                monitoring = vl_metric

        for i in range(self.epochs):

            if self.minibatch_size is not None:
                # shuffling
                tr = tr.sample(frac=1)
                tr_data = tr.iloc[:, :-1]
                tr_label = tr.iloc[:, -1].to_frame()

                for j in range(int(np.ceil(tr.shape[0] / self.minibatch_size))):
                    # if last minibatch is smaller than minibatch_size
                    batch_data = tr_data.iloc[j * self.minibatch_size:(j + 1) * self.minibatch_size - 1, :]
                    batch_label = tr_label.iloc[(j * self.minibatch_size):((j + 1) * self.minibatch_size - 1), :]
                    self.step(batch_data, batch_label)
                # todo add the possibility to use the mean over last n batches
                output = self.feed_forward(tr_data)
                tr_loss.append(self.loss.error(tr_label, output))
                tr_metric.append(self.metric(tr_label, output))
            else:
                output = self.step(tr_data, tr_label)
                tr_loss.append(self.loss.error(tr_label, output))
                tr_metric.append(self.metric(tr_label, output))

            if vl_data is not None:
                output = self.feed_forward(vl_data)
                vl_loss.append(self.loss.error(vl_label, output))
                vl_metric.append(self.metric(vl_label, output))

                if early_stopping:
                    if abs(best_cost - monitoring[i]) < min_delta:
                        last_improvement += 1
                        print("Not improving model (loss) ", last_improvement)
                        if last_improvement == patience:
                            print("No improvement found during the ", last_improvement,
                                  " last iterations, so we stopped!")
                            self.layers = best_model["layers"]
                            self.loss = best_model["loss"]
                            self.optimizer = best_model["optimizer"]
                            break
                    else:
                        best_cost = monitoring[i]
                        last_improvement = 0
                        best_model["layers"] = self.layers
                        best_model["loss"] = self.loss
                        best_model["optimizer"] = self.optimizer
                        print("The epoch ", i, " improved the model ", monitor)

        if vl_data is not None:
            return (tr_metric, tr_loss), (vl_metric, vl_loss)
        return tr_metric, tr_loss

    """

    Using mini-batch → the gradient does not decrease to zero close to a
minimum (as the exact gradient can do)
• Hence fixed learning rate should be avoided:
• For instance, we can decay linearly eta for each step until iteration
    """
