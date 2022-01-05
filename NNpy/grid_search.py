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
            
        Return a NN model with also complete graph topology of the network
    """
    model = NeuralNetwork.init(**nn_params)
    return model


def grid_search_cv(params, train_set, train_label, fit_params=None, n_threads=4, save_path='./', name="grid"):
    """
        Execute Grid Search
        Use multiprocessing library to do a parallel execution

        Param:
            params(dict): dict of params to use for the grid search
            train_set
            train_labels
            fit_params(dict): dict of params to pass to the fit method
            n_threads: number of threads to use
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
        writer.writerow(['average_metric_vl', 'sd_metric_vl', 'average_metric_tr',
                         'network_topology'])

        for item in results:
            writer.writerow([
                str(item['average_metric_vl']),
                str(item['sd_metric_vl']),
                str(item['average_metric_tr']),
                str(item['nn_params'])
            ])
        print(f'Saved results in {file_path}')


