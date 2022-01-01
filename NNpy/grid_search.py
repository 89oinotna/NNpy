import cross_validation as cv
import multiprocessing
import csv
import itertools
import time
import metrics
from network import NeuralNetwork


# Params used to do our GridSearch on our NN model (# of combinations = Cartesian product between params_grid entries)

def run(index, results, nn_params, train_set, train_label, fit_params=None):
    """
        Proxy function where it will start the k_fold cross validation on a configuration
        in an asynchronous way

        Param:
            model(NeuralNetwork): NeuralNetwork object to use
            results(List): List of results obtained in GridSearch
            nn_params(dict): dictionary of param of model object
            Returns nothing but add result from cross validation and nn_params in results list
    """
    print(f"Starting model {index}: {nn_params}")
    avg_metric_vl, sd_vl, avg_metric_tr, res = cv.k_fold_cross_validation(
        nn_params, train_set, train_label, 5, fit_params)
    res = {
        'average_metric_vl': avg_metric_vl,
        'sd_metric_vl': sd_vl,
        'average_metric_tr': avg_metric_tr,
        'nn_params': nn_params,
    }
    print(f"Finished {index}, results are:\n\t{res}")
    results.append(res)


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


def grid_search_cv(params, train_set, train_label, fit_params=None, n_threads=4, save_path='./', name="grid"):
    """
        Execute Grid Search
        Use multiprocessing library to do a parallel execution

        Param:
            save_path(str): string of file path
            name(str): name of the grid search, default will be grid_{hash(params)}

    """
    input_size = train_set.shape[1]

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
    # logging.info(f"RESULTS: {results}")
    start = time.time()

    tasks = []
    combinations = flatten_dict(params)
    print(f"Starting Grid Search: {len(combinations)} * 5 (CV) to try")
    for i, nn_params in enumerate(combinations):
        nn_params['input_size'] = input_size
        tasks.append(pool.apply_async(func=run,
                                      args=(i, results, nn_params, train_set, train_label, fit_params),
                                      ))

    for task in tasks:
        task.get()
    pool.close()
    pool.join()

    # Sort results according to the accuracy of the models
    # if classification => accuracy => higher is better
    # if regression => error => lower is better
    results = list(results)
    results.sort(key=lambda x: x['average_metric_vl'],
                 reverse=True if isinstance(metrics.metric(combinations[0]['metric']), metrics.ClassificationMetric)
                 else False)

    # Write to file results obtained
    if name == 'grid':
        name += f"_{hash(str(params))}"
    save_path += f'{name}.csv'
    write_results(results, save_path)

    with open('./grid_time.txt', 'a') as grid_time:
        total_time = time.gmtime(time.time() - start)
        grid_time.write("Grid Search {} ended in {} hours {} minutes {} seconds \n".format(
            name, total_time.tm_hour, total_time.tm_min, total_time.tm_sec))
    return results


def write_results(results, file_path):
    """
        Write results obtained by the GridSearch into a txt file
        Param:
            file_path(str): path where we will save GridSearch's results
    """

    with open(file_path, 'w') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['average_accuracy_vl', 'sd_accuracy_vl', 'average_metric_tr',
                         'network_topology'])

        for item in results:
            writer.writerow([
                str(item['average_metric_vl']),
                str(item['sd_metric_vl']),
                str(item['average_metric_tr']),
                str(item['nn_params'])
            ])
        print(f'Saved results in {file_path}')


""""""


def final_model():
    """
        Return the final model by bagging the best 10/20 models
    """

    # model params contains the best hyperparameters obtained with the final grid search
    # nn_params = [
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
    # grid_search_cv(params_grid, training_set, len(train_data[0]), len(train_label[0]))
