import numpy as np
import network as nn
import pandas as pd
import activation_functions as af
import losses
import metrics
from input_reading import read_monk, read_cup
import visualization as vis
import grid_search as gs
import itertools
import logging
from sklearn.model_selection import train_test_split

# logging.basicConfig(level=logging.DEBUG)

monk = {'layer_sizes': [4, 1],
                  'act_hidden': 'sigmoid',
                  'act_out': 'sigmoid',
                  'w_init': 'monk',
                  'loss': 'mse',
                  'metric': 'simple_class',
                  'optimizer': {'type_init': 'sgd', 'ETA': 0.6635102010030304, 'ALPHA': 0.8222229742182763,
                                'nesterov': True,
                                'weight_regularizer': {'type_init': 'tikonov',
                                                       'LAMBDA':0.0009853357940222962
                                                       },
                                },
                 'epochs': 500, 'input_size': 17}


cup = {'layer_sizes': (10, 70, 30, 2),
       'act_hidden': 'sigmoid',
       'act_out': 'id',
       'w_init': 'xavier',
       'loss': 'mse',
       'metric': 'mee',
       'optimizer': {
           'type_init': 'sgd',
           'ETA': 0.0625,
           'ALPHA': 0.07,
           'weight_regularizer': {
               'type_init': 'tikonov',
               'LAMBDA': 5.625e-05}
       },
       'batch_size': 32,
       'epochs': 1200,
       'input_size': 10}


# train the best model
# train_data, valid_data, train_labels, valid_labels = train_test_split(
#    train_data, train_label, test_size=0.2)

if __name__ == '__main__':
    #train_data, train_labels, test_data, test_labels = read_cup()  # read_monk("monks-1.train")
    #train_data, valid_data, train_labels, valid_labels = train_test_split(
    #        train_data, train_labels, test_size=0.2)

    train_data, train_labels, test_data, test_labels = read_monk('monks-1')
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=1)

    network = nn.NeuralNetwork.init(**monk)

    (tr_metric, tr_loss), (vl_metric, vl_loss) = network.fit(train_data, train_labels, valid_data, valid_labels)
                                                             #, early_stopping=True, monitor='vl_metric', patience=50)
    vis.plot(tr_loss, vl_loss, tr_metric, vl_metric)
    # grid_results = pd.read_csv("./grid_antonio_1.csv", sep=",")
    # print(grid_results.head())

"""
network = nn.NeuralNetwork.init([4, 1], 17, act_hidden='relu', act_out='sigmoid', w_init='monk', loss='mse',
                                metric='simple_class', optimizer=optimizer, weight_regularizer=wreg, epochs=400)
(tr_metric, tr_loss), (vl_metric, vl_loss) = network.fit(monk_data_tr, monk_label_tr, monk_data_vl, monk_label_vl)
vis.plot(tr_loss, vl_loss, tr_metric, vl_metric)
# x=np.array([[2,1],[2,1]])
# print(np.dot(np.array([2, 2]), x))
"""
