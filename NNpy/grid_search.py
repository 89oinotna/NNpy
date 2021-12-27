import cross_validation as cv
import multiprocessing
import csv
import itertools
import time
from input_reading import read_cup, read_monk
import metrics
from network import NeuralNetwork
import losses
from normalization import normalize, denormalize
from ensembling import Bagging



# Params used to do our GridSearch on our NN model (# of combinations = Cartesian product between params_grid entries)
"""
params_grid = {
    'layer_sizes': [(20, 20), (10, 10), (10, 5, 5), (15, 15), (20,), (30, 20), (50,)],
    'activation_hidden': [act_fun.Tanh, act_fun.Relu, act_fun.LeakyRelu, act_fun.Sigmoid],
    'weight_initialization': [winit.xavier_init, winit.he_init, winit.random_ranged_init],
    'loss': [losses.MeanSquaredError],
    'accuracy': [metrics.MEE],
    'regularization': [0.1, 0.01, 0.001, 0.0001],
    'learning_rates': [0.001, 0.01, 0.1, 0.2, 0.3],
    'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    'batch_size': [5, 10, 20, 40, 60],
    # 'optimizer': [optimizers.SGD],
    'epochs': [1500],
}
"""
params_grid = {
    'layer_sizes': [[5, 1], [8, 1], [10, 1], [12, 1], [14, 1], [16, 1], [17, 1], [19, 1], [22, 1], [25, 1]],
    'activation_hidden': ["relu"],
    'weight_initialization': ["monk"],
    'loss': ["mse"],
    'accuracy': ["simple_class"],
    'regularization': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
    'learning_rates': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85],
    'momentum': [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9],
    'batch_size': [5, 10, 20, 40, 60],
    # 'optimizer': [optimizers.SGD],
    'epochs': [500, 1000, 1500]
}
#train_data, train_label, test_data, test_label = read_cup(training=True, test=False, frac_train=0.8)
train_data, train_label = read_monk("monks-1.train")
test_data, test_label = read_monk("monks-1.test")
training_set = list(zip(train_data, train_label))


def run(model, results, nn_params, dataset):
    """
        Proxy function where it will start the k_fold cross validation on a configuration
        in an asynchronous way

        Param:
            model(NeuralNetwork): NeuralNetwork object to use
            results(List): List of results obtained in GridSearch
            nn_params(dict): dictionary of param of model object
            Returns nothing but add result from cross validation and nn_params in results list
    """
    average_vl, sd_vl, average_tr_error_best_vl, res = cv.k_fold_cross_validation(
        model, dataset, 5)
    print("APPEND   ", average_vl, " APPEND")
    results.append({
        'average_accuracy_vl': average_vl,
        'sd_accuracy_vl': sd_vl,
        'average_tr_error_best_vl': average_tr_error_best_vl,
        'nn_params': nn_params,
    })
    print("Finish {} cross-validation".format(len(results)))


def init_model(nn_params, num_features):
    """
        Create NN model to use to execute a cross validation on it

        Param:
            nn_params(dict): dictionary of params to use to create NN object
            num_features(int): number of features
            output_dim(int): dimension of the output
            
        Return a NN model with also complete graph topology of the network
    """
    print(nn_params)
    nn_params['input_size'] = num_features
    model = NeuralNetwork.init(**nn_params)
    return model


def grid_search_cv(params, dataset, num_features, n_threads=4, save_path='./grid.csv'):
    """
        Execute Grid Search
        Use multiprocessing library to do a parallel execution

        Param:
            save_path(str): string of file path

    """

    def flatten_list(l):
        lst = []
        for i, v in enumerate(l):
            if isinstance(v, dict):
                for sublist in flatten_dict(v):
                    lst.append(sublist)
            else:
                lst.append(v)
        return lst

    def flatten_dict(dic):
        # recursively flattens the dict and return the cartesian product
        for k, v in dic.items():
            if isinstance(v, dict):
                dic[k] = flatten_dict(v)
            if isinstance(v, list):
                dic[k] = flatten_list(v)
        return dict_cart_prod(dic)

    def dict_cart_prod(dic):
        lst = []
        for v in (dict(zip(dic.keys(), x)) for x in itertools.product(*dic.values())):
            lst.append(v)
        return lst

    pool = multiprocessing.Pool(multiprocessing.cpu_count()) if n_threads is None else \
        multiprocessing.Pool(n_threads)
    results = multiprocessing.Manager().list()
    print("RESULTS: ", results)
    start = time.time()

    tasks=[]
    for nn_params in flatten_dict(params):
        model = init_model(nn_params, num_features)
        print("Model:", model)
        tasks.append(pool.apply_async(func=run,
                         args=(model, results, nn_params, dataset),
                        ))

    for task in tasks:
        task.get()
    pool.close()
    pool.join()

    # Sort results according to the accuracy of the models

    # l_results = list(results.sort(key=lambda x: x['average_accuracy_vl'], reverse=True))

    # Write to file results obtained
    write_results(results, save_path)

    with open('./grid_time.txt', 'a') as grid_time:
        total_time = time.gmtime(time.time() - start)
        grid_time.write("Grid Search ended in {} hours {} minutes {} seconds \n".format(
            total_time.tm_hour, total_time.tm_min, total_time.tm_sec))
    return results[0]


def write_results(results, file_path):
    """
        Write results obtained by the GridSearch into a txt file
        Param:
            file_path(str): path where we will save GridSearch's results
    """
    with open(file_path, 'w') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['average_accuracy_vl', 'sd_accuracy_vl', 'average_tr_error_best_vl',
                         'network_topology', 'activation_hidden', 'weight_init', 'regularization',
                         'momentum', 'learning_rate'])

        for item in results:
            writer.writerow([
                str(item['average_accuracy_vl']),
                str(item['sd_accuracy_vl']),
                str(item['average_tr_error_best_vl']),
                str(item['nn_params'][0]),
                str(item['nn_params'][1]),
                str(item['nn_params'][2]),
                str(item['nn_params'][5]),
                str(item['nn_params'][7]),
                str(item['nn_params'][6])
            ])
    return None

    # execute grid search
    # grid_search_cv(parameters, dataset, len(train_data[0]), len(train_label[0]))

""""""
def final_model():
    """
        Return the final model by bagging the best 10/20 models
    """

    # model params contains the best hyperparameters obtained with the final grid search
    #nn_params = [
    """
     BEST 10 or 20 models
     
     List of this type:
     
         [(20, 20), act_fun.Relu, winit.random_ranged_init, losses.MeanSquaredError, 
          metrics.MEE, 0.2, 0.01, 0.1, 10, optimizers.SGD, 1300],

         [(20, 20), act_fun.Relu, winit.random_ranged_init, losses.MeanSquaredError, 
          metrics.MEE, 0.2, 0.01, 0.1, 10, optimizers.SGD, 1300],

         [(20, 20), act_fun.Relu, winit.random_ranged_init, losses.MeanSquaredError, 
          metrics.MEE, 0.2, 0.01, 0.1, 10, optimizers.SGD, 1300], ...

         [(20, 20), act_fun.Relu, winit.random_ranged_init, losses.MeanSquaredError, 
          metrics.MEE, 0.2, 0.01, 0.1, 10, optimizers.SGD, 1300],
            """
#    ]

    # Reading and normalizing data from the ML cup
    # We used 80% of the data for training and the remaining 20% for test
    """train_data, train_label, test_data, test_label = read_cup(
        training=True, test=False, frac_train=0.8)
    train_data, train_label, den_data, den_label = normalize(train_data, train_label)
    test_data, test_label, _, _ = normalize(test_data, test_label, den_data, den_label)

    training_examples = list(zip(train_data, train_label))
    test_examples = list(zip(test_data, test_label))

    model_test = init_model(nn_params[0], len(train_data[0]), 2)
    results = model_test.fit(train_data, train_label, test_data, test_label)
    #TODO: plot accuracy of results

    # create an ensemble object that will contain all the hypothesis
    ensemble = Bagging(len(training_examples))

    # create and add the model to the ensemble

    for params in nn_params:
        network = init_model(params, len(train_data[0]), 2)
        ensemble.add_neural_network(network)

    # training all the models in the ensemble

    ensemble.fit(train_data, train_label, test_examples)

    # check models performance (denormalizing)
    MEE = metrics.MEE()
    i = 1
    for model in ensemble.models:
        predicted_training_data = denormalize(
            model.predict(train_data), den_label)
        error = MEE.error(
            output=predicted_training_data,
            label=denormalize(train_label, den_label)
        )
        print("model ", i, ", training: ", error)

        predicted_test_data = denormalize(
            model.predict(test_data), den_label)
        error = MEE.error(
            output=predicted_test_data,
            label=denormalize(test_label, den_label)
        )

        print("model ", i, ", test: ", error)
        i += 1

    # check ensemble performance

    predicted_training_data = denormalize(
        ensemble.predict(train_data), den_label)
    error = MEE.error(
        output=predicted_training_data,
        label=denormalize(train_label, den_label)
    )
    print("ensemble training: ", error)

    predicted_test_data = denormalize(
        ensemble.predict(test_data), den_label)
    error = MEE.error(
        output=predicted_test_data,
        label=denormalize(test_label, den_label)
    )

    print("ensemble test: ", error)

    return ensemble


final_model()
"""
    #grid_search_cv(params_grid, training_set, len(train_data[0]), len(train_label[0]))
