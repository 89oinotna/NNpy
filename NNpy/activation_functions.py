import numpy as np

# The activation functions developed are:

#   - Identity function
#   - ReLU function
#   - Leaky ReLU function
#   - ELU function
#   - Sigmoid function
#   - Tanh function

# For each of them have been developed even the derivative


'''
    The class ActivationFunction is made for the activation functions.
    It takes as input:
        - name: name of the activation function
        - func: values of the activation function given a list 
'''


class ActivationFunction:

    def __init__(self, name, func):
        self.__func = func
        self.__name = name

    def name(self):
        return self.__name

    def func(self):
        return self.__func


'''
    The class DerivationActivationFunction is made for the derivative of activation functions
    and it is a subclass of ActivationFunction.
    
    It takes as input:
        - name: name of the activation function
        - func: values of the activation function given a list 
        - deriv: values of the derivative of the activation function given a list
'''


class DerivationActivationFunction(ActivationFunction):

    def __init__(self, name, func, deriv):
        super(DerivationActivationFunction, self).__init__(name=name, func=func)
        self.__deriv = deriv

    def deriv(self):
        return self.__deriv


'''
    The function identity_function takes in input x that it is a list and compute the identity function.
    Output:
        - a list x that it's exactly the list that it took as input 
'''


def identity_function(x):
    return x


'''
    The function identity_deriv takes in input x that it is a list and compute the derivative.
    Output:
        - a list x that it's composed by all 1s.
'''


def identity_deriv(x):
    der = [1.] * len(x)
    return der


'''
    The function relu_function takes in input x that it is a list.
    Output:
        - a list x s.t. for each element inside we choose the maximum between i and 0
'''


def relu_function(x):
    return [np.maximum(0, i) for i in x]


'''
    The function relu_deriv takes in input x that it is a list and compute the derivative.
    Output:
        - a list x that it's composed by 0 if the value is <= 0, 1 otherwise.
'''


def relu_deriv(x):
    return [0 if i <= 0 else 1 for i in x]


'''
    The function leaky_function takes in input x that it is a list.
    Output:
        - a list x s.t. for each element inside we choose the maximum between 0.01*i and i
'''


def leaky_function(x):
    return [np.maximum(0.01*i, i) for i in x]


'''
    The function leaky_deriv takes in input x that it is a list and compute the derivative.
    Output:
        - a list x that it's composed by 0.01 if the value is <= 0, 1 otherwise.
'''


def leaky_deriv(x):
    return [0.01 if i <= 0 else 1 for i in x]


'''
    The function elu_function takes in input x that it is a list and alpha that is a scalar.
    The default value for alpha is 0.01.
    Output:
        - a list x s.t. for each element inside we compute alpha * (e^i - 1)
'''


def elu_function(x, alpha=0.01):
    return [i if i > 0 else np.multiply(alpha, np.subtract(np.exp(i), 1)) for i in x]


'''
    The function elu_deriv takes in input x that it is a list and alpha that is a scalar and compute the derivative.
    The default value for alpha is 0.01.
    Output:
        - a list x that it's composed by 1 if the value is > 0, otherwise 
        (the value of elu function with the same alpha + alpha).
'''


def elu_deriv(x, alpha=0.01):
    elu_values = elu_function(x, alpha)
    j = 0
    res = []
    for i in x:
        if i > 0:
            res.append(1)
        else:
            res.append(np.add(elu_values[j], alpha))
        j += 1
    return res


'''
    The function sigmoid_function takes in input x that it is a list.
    Output:
        - a list x s.t. for each element inside x we compute 1 / (1 + e^-i)
'''


def sigmoid_function(x):
    num = [1.] * len(x)
    den = [np.add(1, np.exp(-i)) for i in x]
    return np.divide(num, den)


'''
    The function sigmoid_deriv takes in input x that it is a list and compute the derivative.
    Output:
        - a list x s.t. for each element we have: the value of sigmoid function * (1- the value of sigmoid function).
'''


def sigmoid_deriv(x):
    fx = sigmoid_function(x)
    return np.multiply(fx, (np.subtract(1, fx)))


'''
    The function tanh_function takes in input x that it is a list.
    Output:
        - a list x s.t. for each element inside x we compute the tanh value
'''


def tanh_function(x):
    return np.tan(x)


'''
    The function tanh_deriv takes in input x that it is a list and compute the derivative.
    Output:
        - a list x s.t. for each v we have (1 - tanh(v)^2).
'''


def tanh_deriv(x):
    return np.subtract(1, np.power(np.tan(x), 2))
