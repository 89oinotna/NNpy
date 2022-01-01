import numpy as np
import math

from NNpy.network import NeuralNetwork
from normalization import denormalize
import metrics
import copy
import logging


def fold_i_of_k(dataset, i, k):
    n = len(dataset)
    return dataset[n * (i - 1) // k:n * i // k]


def split(dataset, num_subsets):
    """return the indices into which split the dataset into num_subsets subsets

    Args:
        dataset (list): dataset to be split
        num_subsets (int): number of subsets into which split the dataset

    Returns:
        list of tuple: indices into which split the dataset into num_subsets subsets.
        For instance, with k = 5 and a dataset of 500 patterns, the method returns
        ((0,100),(100, 200),(200,300),(300,400),(400,500)).
    """
    num_elements_per_set = math.floor(len(dataset) / num_subsets)

    return [(i * num_elements_per_set, (
            i + 1) * num_elements_per_set) for i in range(0, num_subsets)]


def init_model(nn_params):
    """
        Create NN model to use to execute a cross validation on it

        Param:
            nn_params(dict): dictionary of params to use to create NN object
            num_features(int): number of features
            output_dim(int): dimension of the output

        Return a NN model with also complete graph topology of the network
    """
    model = NeuralNetwork.init(**nn_params)
    return model


def k_fold_cross_validation(model, train_set, train_label, n_folds, fit_params=None):
    """cross validation implementation

    Args:
        fit_params: parameters for the fit method of the model
        model (NeuralNetwork): neural network from each fold iteration start
        training_set (array of tuple): data for training
        n_folds (int): number of folds
        den_label ((float, float), optional): tuple of the form (mean, variance) used for denormalization.
        Defaults to None. If not indicated, cross-validation does not perform any
        denormalization.

    Returns:
        (float64, float64, float64, array):
            * mean validation error
            * standard deviation over the validation error
            * the mean training error when the validation error was minimum
            * list of all the results
    """
    # output of the cross validation
    metric_res = {'tr': np.zeros(n_folds),
                  'vl': np.zeros(n_folds)}
    results = []

    # get the indexes to break down the data set into the different folds
    splitted_dataset_indices = split(train_set, n_folds)
    for k in range(0, n_folds):
        """# create a deep copy of the model passed as argument
        model_k = copy.deepcopy(model)"""
        model_k = init_model(model)
        # dividing training and validation set
        training_set = np.delete(train_set, np.r_[splitted_dataset_indices[k][0]:splitted_dataset_indices[k][1]],
                                 axis=0)
        validation_set = train_set[np.r_[splitted_dataset_indices[k][0]:splitted_dataset_indices[k][1]]]
        training_label = np.delete(train_label, np.r_[splitted_dataset_indices[k][0]:splitted_dataset_indices[k][1]],
                                   axis=0)
        validation_label = train_label[np.r_[splitted_dataset_indices[k][0]:splitted_dataset_indices[k][1]]]

        # train the model
        if fit_params is None:
            (tr_metric, tr_loss), (vl_metric, vl_loss) = model_k.fit(training_set, training_label, validation_set,
                                                                     validation_label)
        else:
            (tr_metric, tr_loss), (vl_metric, vl_loss) = model_k.fit(training_set, training_label, validation_set,
                                                                     validation_label, **fit_params)
        logging.debug("Finished for k = {}".format(k))

        metric_res['vl'][k] = vl_metric[-1]
        metric_res['tr'][k] = tr_metric[-1]

        results.append([(tr_metric, tr_loss), (vl_metric, vl_loss)])

        # TODO accuracy plot:

    return np.round(np.mean(metric_res['vl']), 10), np.round(np.std(metric_res['vl']), 10), \
           np.round(np.mean(metric_res['tr']), 10), results
