import numpy as np


# The activation functions developed are:

#   - Identity function
#   - ReLU function
#   - Leaky ReLU function
#   - ELU function
#   - Sigmoid function
#   - Tanh function

# For each of them have been developed even the derivative

def activation_function(type_init, **kwargs):
    init = {
        'id': lambda **kwargs: Identity(),
        'relu': lambda **kwargs: Relu(),
        'leaky': lambda **kwargs:LeakyRelu,
        'elu': lambda **kwargs:Elu(),
        'sigmoid': lambda **kwargs:Sigmoid(),
        'tanh': lambda **kwargs:Tanh()
    }
    matrix = init[type_init](**kwargs)
    return matrix


class ActivationFunction:
    """
        The class ActivationFunction is an interface to define activation functions
        with their derivative
    """

    def output(self, x):
        pass

    def derivative(self, x):
        pass


class Identity(ActivationFunction):

    def output(self, x):
        """
        The function identity_function takes in input x that it is a list and compute the identity function.
        :param x: list to compute
        :return: a list x that it's exactly the list that it took as input
        """
        return x

    def derivative(self, x):
        """
        The function identity_deriv takes in input x that it is a list and compute the derivative.
        :param x: list to compute
        :return: a list x that it's composed by all 1s.
        """
        return 1


class Relu(ActivationFunction):
    def output(self, x):
        """
        The function relu_function takes in input x that it is a list.
        Output:
            - a list x s.t. for each element inside we choose the maximum between i and 0
        :param self:
        :param x:
        :return:
        """
        return np.maximum(x, 0)

    def derivative(self, x):
        """
          The function relu_deriv takes in input x that it is a list and compute the derivative.
        Output:
            - a list x that it's composed by 0 if the value is <= 0, 1 otherwise.
        :param x:
        :return:
        """
        return np.greater(x, 0)


class LeakyRelu(ActivationFunction):
    def output(self, x):
        """
            The function leaky_function takes in input x that it is a list.
    Output:
        - a list x s.t. for each element inside we choose the maximum between 0.01*i and i
        :return:
        """
        return np.maximum(0.01 * x, x)

    def derivative(self, x):
        """
        The function leaky_deriv takes in input x that it is a list and compute the derivative.

        Output:
        - a list x that it's composed by 0.01 if the value is <= 0, 1 otherwise.
        :param self:
        :param x:
        :return:
        """
        return np.where(x >= 0, 1, 0.01)


class Elu(ActivationFunction):

    def __init__(self, alpha=0.01):
        """
        :param alpha: scalar, default = 0.01
        """
        self.alpha = alpha

    def output(self, x):
        """
        The function elu_function takes in input x that it is a list.
        Output:
            - a list x s.t. for each element inside we compute alpha * (e^i - 1)
        :param self:
        :param x:
        :return:
        """
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1.))

    def derivative(self, x):
        """
        The function elu_deriv takes in input x that it is a list and compute the derivative.
        Output:
            - a list x that it's composed by 1 if the value is > 0, otherwise
            (the value of elu function with the same alpha + alpha).
        :param x:
        :return:
        """
        return np.where(x >= 0, 1, self.output(x) + self.alpha)


class Sigmoid(ActivationFunction):

    def output(self, x):
        """
        The function sigmoid_function takes in input x that it is a list.
        Output:
        - a list x s.t. for each element inside x we compute 1 / (1 + e^-i)
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """
        The function sigmoid_deriv takes in input x that it is a list and compute the derivative.
        Output:
            - a list x s.t. for each element we have: the value of sigmoid function * (1- the value of sigmoid function).
        :param x:
        :return:
        """
        fx = self.output(x)
        return fx * (1 - fx)


class Tanh(ActivationFunction):

    def output(self, x):
        """
        The function tanh_function takes in input x that it is a list.
        Output:
            - a list x s.t. for each element inside x we compute the tanh value
        :param self:
        :param x:
        :return:
        """
        return np.tanh(x)

    def derivative(self, x):
        """
        The function tanh_deriv takes in input x that it is a list and compute the derivative.

        Output:
        - a list x s.t. for each v we have (1 - tanh(v)^2).
        :param self:
        :param x:
        :return:
        """
        return np.subtract(1, np.power(np.tanh(x), 2))
