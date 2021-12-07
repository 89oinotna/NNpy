import numpy as np


# The activation functions developed are:

#   - Identity function
#   - ReLU function
#   - Leaky ReLU function
#   - ELU function
#   - Sigmoid function
#   - Tanh function

# For each of them have been developed even the derivative


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
        der = [1.] * len(x)
        return der


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
        return [np.maximum(0, i) for i in x]

    def derivative(self, x):
        """
          The function relu_deriv takes in input x that it is a list and compute the derivative.
        Output:
            - a list x that it's composed by 0 if the value is <= 0, 1 otherwise.
        :param x:
        :return:
        """
        return [0 if i <= 0 else 1 for i in x]


class LeakyRelu(ActivationFunction):
    def output(self, x):
        """
            The function leaky_function takes in input x that it is a list.
    Output:
        - a list x s.t. for each element inside we choose the maximum between 0.01*i and i
        :return:
        """
        return [np.maximum(0.01 * i, i) for i in x]

    def derivative(self, x):
        """
        The function leaky_deriv takes in input x that it is a list and compute the derivative.
    Output:
        - a list x that it's composed by 0.01 if the value is <= 0, 1 otherwise.
        :param self:
        :param x:
        :return:
        """
        return [0.01 if i <= 0 else 1 for i in x]


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
        return [i if i > 0 else np.multiply(self.alpha, np.subtract(np.exp(i), 1)) for i in x]

    def derivative(self, x):
        """
        The function elu_deriv takes in input x that it is a list and compute the derivative.
        Output:
            - a list x that it's composed by 1 if the value is > 0, otherwise
            (the value of elu function with the same alpha + alpha).
        :param x:
        :return:
        """
        elu_values = self.output(x)
        j = 0
        res = []
        for i in x:
            if i > 0:
                res.append(1)
            else:
                res.append(np.add(elu_values[j], self.alpha))
            j += 1
        return res


class Sigmoid(ActivationFunction):

    def output(self, x):
        """
        The function sigmoid_function takes in input x that it is a list.
        Output:
        - a list x s.t. for each element inside x we compute 1 / (1 + e^-i)
        :param x:
        :return:
        """
        num = [1.] * len(x)
        den = [np.add(1, np.exp(-i)) for i in x]
        return np.divide(num, den)

    def derivative(self, x):
        """
        The function sigmoid_deriv takes in input x that it is a list and compute the derivative.
        Output:
            - a list x s.t. for each element we have: the value of sigmoid function * (1- the value of sigmoid function).
        :param x:
        :return:
        """
        fx = self.output(x)
        return np.multiply(fx, (np.subtract(1, fx)))


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
        return np.tan(x)

    def derivative(self, x):
        """
        The function tanh_deriv takes in input x that it is a list and compute the derivative.

        Output:
        - a list x s.t. for each v we have (1 - tanh(v)^2).
        :param self:
        :param x:
        :return:
        """
        return np.subtract(1, np.power(np.tan(x), 2))