import numpy as np
import network as nn
import pandas as pd
# print(wi.xavier_init(2,2)[0:, 1:])
import activation_functions as af
import losses
import metrics
from input_reading import read_monk, read_cup
import visualization as vis
import grid_search as gs
import itertools
import logging
from sklearn.model_selection import train_test_split

#logging.basicConfig(level=logging.DEBUG)
monk_data_tr, monk_tr_label, monk_test_data, monk_test_label = read_monk('monks-1')
monk_data_tr, monk_valid_data, monk_tr_label, monk_valid_labels = train_test_split(
    monk_data_tr, monk_tr_label, test_size=0.2, random_state=1)


optimizer = {
    'type_init': 'sgd',
    # 'nesterov': True,
    'ETA': 0.8,
    #'momentum_window': 1,
    'ALPHA': 0.8,
    #'weight_regularizer': {
    #    'type_init': 'tikonov',
    #    'LAMBDA': 0.1
    #}
}

'''
wreg = {'type_init': 'tikonov', 'LAMBDA': 0.001}
'''


params = {
    'layer_sizes': [(15, 1)],
    'act_hidden': ['relu', 'sigmoid'],
    'act_out': ["sigmoid"],
    'w_init': ["monk"],
    'loss': ["mse"],
    'metric': ["simple_class"],

    'optimizer': {
        'type_init': ['sgd'],
        'nesterov': [True],
        'ETA': [0.2, 0.4, 0.6, 0.8],
        #'momentum_window': 1,
        'ALPHA': [0.2, 0.4, 0.6, 0.8],
        'weight_regularizer': {
            'type_init': ['tikonov'],
            'LAMBDA': [0.01, 0.001, 0.0001, 0.00001]
        }
    },
    'epochs': [500]
}


# train the best model
# train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_label, test_size=0.2)

if __name__ == '__main__':

    #grid_results = gs.grid_search_cv(params, monk_data_tr, monk_tr_label)

    minimum = 10
    network = nn.NeuralNetwork.init([15, 1], 17, act_hidden='relu', act_out='sigmoid', w_init='random_ranged', loss='mse',
                            metric='simple_class', optimizer=optimizer, epochs=500)

    for i in range(200):
        (tr_metric, tr_loss), (vl_metric, vl_loss) = network.fit(monk_data_tr, monk_tr_label, monk_test_data,
                                                             monk_test_label)
        if (vl_loss[-1]) < minimum:
            minimum = vl_loss[-1]
            best_tr_loss = tr_loss
            best_vl_loss = vl_loss
            best_tr_metric = tr_metric
            best_vl_metric = vl_metric
            print(f'Test {network.evaluate(monk_test_data, monk_test_label)}')


    vis.plot(best_tr_loss, best_vl_loss, best_tr_metric, best_vl_metric)
    # x=np.array([[2,1],[2,1]])
    # print(np.dot(np.array([2, 2]), x))

