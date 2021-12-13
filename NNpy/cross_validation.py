import numpy as np
import math
from normalization import denormalize
import metrics
#import network
#import layer
#import weights_init as winit
#import activation_functions as act_fun


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


def k_fold_cross_validation(model, training_set, n_folds, den_label=None):
    """cross validation implementation

    Args:
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
    MEE = metrics.MEE
    sum_tr_errors = 0
    best_vl_err = np.inf
    tr_err_with_best_vl_error = np.inf
    errors = np.zeros(n_folds)
    results = []

    # get the indexes to break down the data set into the different folds
    splitted_dataset_indices = split(training_set, n_folds)
    for k in range(0, n_folds):
        # create a deep copy of the model passed as argument
        model_k = model.deepcopy()
        # dividing training and validation set
        training_set = training_set[:splitted_dataset_indices[k]
        [0]] + training_set[splitted_dataset_indices[k][1]:]
        validation_set = training_set[splitted_dataset_indices[k]
                                 [0]:splitted_dataset_indices[k][1]]

        # train the model
        (tr_metric, tr_loss), (vl_metric, vl_loss) = model_k.fit(training_set, validation_set)
        print("Finished for k = {}".format(k))
        if vl_loss < best_vl_err:
            tr_err_with_best_vl_error = tr_loss

        # update things for the cross validation result

        # sum the training error that we have when we reach the minimum validation error

        sum_tr_errors += tr_err_with_best_vl_error

        inputs_validation = np.array([elem[0]
                                      for elem in validation_set])
        targets_validation = np.array([elem[1]
                                       for elem in validation_set])

        # add the error to the vector errors to compute the standard deviation and the mean accuracy
        error = 0

        # if den_label, then the result predicted by the hypothesis is denormalized
        # as for the targets in the validation set
        if den_label:
            predicted_test_data = denormalize(
                model_k.predict(inputs_validation), den_label)
            error = MEE.error(
                output=predicted_test_data,
                label=denormalize(targets_validation, den_label)
            )
        else:
            error = vl_metric

        errors[n_folds] = error

        results.append([(tr_metric, tr_loss), (vl_metric, vl_loss)])

        #TODO accuracy plot:

    return np.round(np.mean(errors), 10), np.round(np.std(errors), 10), np.round(
        sum_tr_errors / n_folds, 10), results
