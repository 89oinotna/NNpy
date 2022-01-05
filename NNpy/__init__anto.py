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

# logging.basicConfig(level=logging.DEBUG)
monk_data_tr, monk_tr_label, monk_test_data, monk_test_label = read_monk('monks-1')
monk_data_tr, monk_valid_data, monk_tr_label, monk_valid_labels = train_test_split(
    monk_data_tr, monk_tr_label, test_size=0.2, random_state=1)

cup = {'layer_sizes': [(20, 40, 20, 2), (40, 40, 2)],
       'act_hidden': ['sigmoid', 'tanh', 'relu'],
       'act_out': ['id'],
       'w_init': ['xavier'],
       'loss': ['mse'],
       'metric': ['mee'],
       'optimizer': {
           'type_init': ['sgd'],
           'ETA': [0.01, 0.1, 0.2],
           'ALPHA': [0, 0.1, 0.2, 0.4],
           'momentum_window': [40],
           'weight_regularizer': {
               'type_init': ['tikonov'],
               'LAMBDA': [0, 0.0001, 0.001]
           },
           'variable_eta': {
               'eta_tau': [0.001],
               'tau': [200],
           }
       },
       'minibatch_size': [32],
       'epochs': [500]
       }

monk={'layer_sizes': [(15, 1)],
       'act_hidden': ['sigmoid', 'tanh', 'relu'],
       'act_out': ['sigmoid'],
       'w_init': ['monk'],
       'loss': ['mse'],
       'metric': ['simple_class'],
       'optimizer': {
           'type_init': ['sgd'],
           'nesterov':[False],
           'ETA': list(np.linspace(0.1, 0.9, 9)),
           'ALPHA': list(np.linspace(0.1, 0.9, 9)),
           #'momentum_window': [40],

           #'variable_eta': {
           #    'tau': [0.001],
           #    'step': [200],
           #}
       },
       #'minibatch_size': [32],
       'epochs': [500]
       }


# train the best model
# train_data, valid_data, train_labels, valid_labels = train_test_split(
#    train_data, train_label, test_size=0.2)

if __name__ == '__main__':
    #train_data, train_labels, test_data, test_labels = read_cup(frac_train=0.8)  # read_monk("monks-1.train")
    #train_data, valid_data, train_labels, valid_labels = train_test_split(
    #    train_data, train_labels, test_size=0.2)

    train_data, train_labels, test_data, test_labels = read_monk("monks-2")

    grid_results = gs.grid_search_cv(monk, train_data, train_labels)

    # grid_results = pd.read_csv("datasets/gs_results/grid_antonio_2.csv", sep=",")

    # network = nn.NeuralNetwork.init(**eval(grid_results.loc[0, 'network_topology']), minibatch_size=2)
    # print(grid_results.loc[0, 'network_topology'])

    # (tr_metric, tr_loss), (vl_metric, vl_loss) = network.fit(train_data, train_labels, valid_data, valid_labels, early_stopping=True)
    # vis.plot(tr_loss, vl_loss, tr_metric, vl_metric)
    # network.save()

    # network = nn.NeuralNetwork.load(name='144670304208')
    # print(network.evaluate(valid_data, valid_labels))

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
