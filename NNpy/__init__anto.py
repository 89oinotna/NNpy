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
monk_data_tr, monk_tr_label, monk_test_data, monk_test_label= read_monk('monks-1')
monk_data_tr, monk_valid_data, monk_tr_label, monk_valid_labels = train_test_split(
    monk_data_tr, monk_tr_label, test_size=0.2, random_state=1)

optimizer = {
    'type_init': 'sgd',
    'nesterov':True,
    'ETA':0.5,
    #'momentum_window': 1,
    'ALPHA':0.8
}
wreg ={'type_init': 'tikonov', 'LAMBDA': 0.001}


params = {
    'layer_sizes': [(10, 70, 30, 2)],
    'act_hidden': ['sigmoid'],
    'act_out': ["id"],
    'w_init': ["xavier"],
    'loss': ["mse"],
    'metric': ["simple_accuracy"],

    'optimizer': {
        'type_init': ['sgd'],
        'nesterov':[True, False],
        'ETA':np.random.uniform(0,0.08,10),#np.linspace(0.055, 0.076, 5),#[0.001, 0.01, 0.1, 0.2],
        #'momentum_window': 1,
        'ALPHA': np.random.uniform(0.6,0.85,8),#np.linspace(0.60, 0.85, 5),
        'weight_regularizer': {'type_init':['tikonov'],
                              'LAMBDA':np.random.uniform(0, 7.5e-05, 8)#np.linspace(0, 7.5e-05, 6)
                              },
    },
    'batch_size': [32],
    'epochs': [500]
}

monk_params={
'layer_sizes': [(4,1), (2, 2, 1)],
    'act_hidden': ['sigmoid', 'tanh', 'relu'],
    'act_out': ["sigmoid"],
    'w_init': ["monk"],
    'loss': ["mse"],
    'metric': ["simple_class"],
    'optimizer': {
        'type_init': ['sgd'],
        'nesterov':[True, False],
        'ETA':np.random.uniform(0.5,0.9,5),#np.linspace(0.055, 0.076, 5),#[0.001, 0.01, 0.1, 0.2],
        'ALPHA': np.random.uniform(0.5,0.9,5),  # np.random.uniform(0.0,0.9,8),#np.linspace(0.60, 0.85, 5),
        'weight_regularizer': {'type_init':['tikonov'],
                              'LAMBDA':np.random.uniform(0.0,0.001,5),
                       # np.random.uniform(0, 0.1, 8)#np.linspace(0, 7.5e-05, 6)
                              },
    },
    'epochs': [1000]
}


minibatch_monk = {'layer_sizes': (8,1),
             'act_hidden': 'sigmoid',
             'act_out': 'id',
             'w_init': 'monk',
             'loss': 'mse',
             'metric': 'mee',
             'optimizer': {'type_init': 'sgd', 'ETA': 0.3, 'ALPHA': 0.8,
                           'weight_regularizer': {'type_init': 'tikonov',
                                                  'LAMBDA': 0.0001
                                                  },
                           },
             'batch_size': 32, 'epochs': 1000, 'input_size': 10}

cup = {'layer_sizes': (10, 70, 30, 2),
       'act_hidden': 'sigmoid',
       'act_out': 'id',
       'w_init': 'xavier',
       'loss': 'mse',
       'metric': 'mee',
       'optimizer': {
           'type_init': 'sgd',
           'ETA': 0.1,
           'ALPHA': 0.2,
           'weight_regularizer': {
               'type_init': 'tikonov',
               'LAMBDA': 1e-05}
       },
       'batch_size': 32,
       'epochs': 500,
       'input_size': 10}


# train the best model
#train_data, valid_data, train_labels, valid_labels = train_test_split(
#    train_data, train_label, test_size=0.2)

if __name__ == '__main__':
    #train_data, train_labels, test_data, test_labels = read_cup(frac_train=0.8)  # read_monk("monks-1.train")

    grid_results = gs.grid_search_cv(monk_params, monk_data_tr, monk_tr_label)

    #grid_results = pd.read_csv("datasets/gs_results/grid_antonio_2.csv", sep=",")

    #network = nn.NeuralNetwork.init(**eval(grid_results.loc[0, 'network_topology']))
    #(tr_metric, tr_loss), (vl_metric, vl_loss) = network.fit(train_data, train_labels, valid_data, valid_labels, early_stopping=True)
    #vis.plot(tr_loss, vl_loss, tr_metric, vl_metric)

    #grid_results = pd.read_csv("./grid_antonio_1.csv", sep=",")
    #print(grid_results.head())

"""
network = nn.NeuralNetwork.init([4, 1], 17, act_hidden='relu', act_out='sigmoid', w_init='monk', loss='mse',
                                metric='simple_class', optimizer=optimizer, weight_regularizer=wreg, epochs=400)
(tr_metric, tr_loss), (vl_metric, vl_loss) = network.fit(monk_data_tr, monk_label_tr, monk_data_vl, monk_label_vl)
vis.plot(tr_loss, vl_loss, tr_metric, vl_metric)
# x=np.array([[2,1],[2,1]])
# print(np.dot(np.array([2, 2]), x))
"""
