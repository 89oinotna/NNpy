import numpy as np


def weights_init(type_init, **kwargs):
    init = {
        'random ranged': random_ranged_init,
        'xavier': xavier_init,
        'he': he_init
    }
    matrix_weights = init[type_init](**kwargs)
    return matrix_weights


'''
    The random_ranged_init allows to initialize the weights randomly in range.
    
    The input parameters are:
        - num_unit -> # of units such that their input weights have to be initialize
        - num_input -> # of inputs for each unit 
        - range -> range in which the random weights have to be generated. The default
            value is [-0.7,0.7]
        
    The output parameters are:
        - matrix_weights -> a matrix with dimensions num_unit x num_input. In other words
            for each row there are the input weights for a specific unit.
            The first column is made by the bias weights.
'''


def random_ranged_init(num_unit, num_input, range=(-0.7, 0.7)):
    min_range, max_range = range[0], range[1]
    if min_range > max_range:
        raise ValueError('The min value must be <= than the max value')
    matrix_weights = np.random.uniform(low=min_range, high=max_range, size=(num_unit, num_input+1))

    return matrix_weights


'''
    The xavier_init allows to initialize the weights following the Xavier initialization.
    The bias weights are set to 0 for all the units and the other weights are obtained
    generating random numbers and multiply them by square root of (1/number of input)

    The input parameters are:
        - num_unit -> # of units such that their input weights have to be initialize
        - num_input -> # of inputs for each unit 

    The output parameters are:
        - matrix_weights -> a matrix with dimensions num_unit x num_input. In other words
            for each row there are the input weights for a specific unit.
            The first column is made by the bias weights, in fact is made by all 0s.
    
    It has been proved that this initialization performs well when the activation function
    of the neurons is the sigmoid or the tanh.
'''


def xavier_init(num_unit, num_input):
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.randn(num_unit, num_input) * np.sqrt(1 / num_input)
    matrix_weights = np.concatenate((bias_weights, input_weights), axis=1)
    return matrix_weights


'''
    The he_init allows to initialize the weights following the He initialization.
    The bias weights are set to 0 for all the units and the other weights are obtained
    generating random numbers and multiply them by square root of (2/number of input)

    The input parameters are:
        - num_unit -> # of units such that their input weights have to be initialize
        - num_input -> # of inputs for each unit

    The output parameters are:
        - matrix_weights -> a matrix with dimensions num_unit x num_input. In other words
            for each row there are the input weights for a specific unit.
            The first column is made by the bias weights, in fact is made by all 0s.

    It has been proved that this initialization performs well when the activation function
    of the neurons is the ReLU.
'''


def he_init(num_unit, num_input):
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.randn(num_unit, num_input) * np.sqrt(2 / num_input)
    matrix_weights = np.concatenate((bias_weights, input_weights), axis=1)
    return matrix_weights
