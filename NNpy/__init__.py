import numpy as np
import network as nn
import pandas as pd
# print(wi.xavier_init(2,2)[0:, 1:])
import activation_functions as af
import losses
import metrics
from input_reading import read_monk
import visualization as vis
import grid_search as gs
import itertools
import logging
from sklearn.model_selection import train_test_split

#logging.basicConfig(level=logging.DEBUG)

monk_data, monk_label = read_monk('monks-1.test')
monk_data_tr, monk_data_vl = np.array_split(monk_data,2)
monk_label_tr, monk_label_vl = np.array_split(monk_label,2)
#print(monk_data[:1000, :])
optimizer = {
    'type_init': 'sgd',
    'nesterov':True,
    'ETA':0.5,
    #'momentum_window': 1,
    'ALPHA':0.8
}
wreg ={'type_init': 'tikonov', 'LAMBDA': 0.001}


params = {
    'layer_sizes': [(5, 1), (8, 1)],
    'act_hidden': ["relu"],
    'act_out': ["sigmoid"],
    'w_init': ["monk"],
    'loss': ["mse"],
    'metric': ["simple_class"],
    'weight_regularizer': {'type_init':['tikonov'],
                              'LAMBDA':[0.1, 0.01, 0.001, 0.0001, 0.00001]
                              },
    'optimizer': {
        'type_init': ['sgd'],
        #'nesterov':[True],
        'ETA':[0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8],
        #'momentum_window': 1,
        'ALPHA': [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
    },
    'batch_size': [32, 64],
    'epochs': [500]
}

minibatch = {'layer_sizes': (8, 1), 'act_hidden': 'relu', 'act_out': 'sigmoid', 'w_init': 'monk', 'loss': 'mse', 'metric': 'simple_class', 'weight_regularizer': {'type_init': 'tikonov', 'LAMBDA': 0.001}, 'optimizer': {'type_init': 'sgd', 'ETA': 0.7, 'ALPHA': 0.8, 'weight_regularizer': {'type_init': 'tikonov', 'LAMBDA': 0.001}}, 'batch_size': 32, 'epochs': 500, 'input_size': 17}

train_data, train_label = read_monk("monks-1.train")
test_data, test_label = read_monk("monks-1.test")

# train the best model
train_data, valid_data, train_labels, valid_labels = train_test_split(
    train_data, train_label, test_size=0.2)

if __name__ == '__main__':
    #gs.grid_search_cv(params, train_data, train_label)
    network = nn.NeuralNetwork.init(**minibatch)

    (tr_metric, tr_loss), (vl_metric, vl_loss) = network.fit(train_data, train_labels, valid_data, valid_labels)
    vis.plot(tr_loss, vl_loss, tr_metric, vl_metric)

"""
network = nn.NeuralNetwork.init([4, 1], 17, act_hidden='relu', act_out='sigmoid', w_init='monk', loss='mse',
                                metric='simple_class', optimizer=optimizer, weight_regularizer=wreg, epochs=400)
(tr_metric, tr_loss), (vl_metric, vl_loss) = network.fit(monk_data_tr, monk_label_tr, monk_data_vl, monk_label_vl)
vis.plot(tr_loss, vl_loss, tr_metric, vl_metric)
# x=np.array([[2,1],[2,1]])
# print(np.dot(np.array([2, 2]), x))
"""
