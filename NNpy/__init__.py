import numpy as np
import network as nn
# print(wi.xavier_init(2,2)[0:, 1:])
import activation_functions as af
import losses
import metrics
from input_reading import read_monk
import visualization as vis
import grid_search as gs

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

'''
params = {
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
}'''
params = {
    'layer_sizes': [(5, 1), (8, 1), (10, 1), (12, 1), (14, 1), (16, 1)],
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
train_data, train_label = read_monk("monks-1.train")
test_data, test_label = read_monk("monks-1.test")
training_set = list(zip(train_data, train_label))

if __name__ == '__main__':
    gs.grid_search_cv(params, training_set, len(train_data[0]), len(train_label[0]))
"""
network = nn.NeuralNetwork.init([4, 1], 17, act_hidden='relu', act_out='sigmoid', w_init='monk', loss='mse',
                                metric='simple_class', optimizer=optimizer, weight_regularizer=wreg, epochs=400)
(tr_metric, tr_loss), (vl_metric, vl_loss) = network.fit(monk_data_tr, monk_label_tr, monk_data_vl, monk_label_vl)
vis.plot(tr_loss, vl_loss, tr_metric, vl_metric)
# x=np.array([[2,1],[2,1]])
# print(np.dot(np.array([2, 2]), x))
"""
