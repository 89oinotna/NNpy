import numpy as np
import network as nn
# print(wi.xavier_init(2,2)[0:, 1:])
import activation_functions as af
import losses
import metrics
from input_reading import read_monk
import visualization as vis

monk_data, monk_label = read_monk('monks-1.test')
# print(np.array(monk_data))
network = nn.NeuralNetwork.init([4, 1], 17, act_hidden='relu', act_out='sigmoid', w_init='monk', loss='mse',
                                metric='simple_class', optimizer='sgd', epochs=400, ETA=0.5,
                                weight_regularizer='tikonov', ALPHA=0.8, LAMBDA=0.001, nesterov=True)
metric, loss = network.fit(monk_data, monk_label)
vis.plot(loss, loss, metric, metric)
# x=np.array([[2,1],[2,1]])
# print(np.dot(np.array([2, 2]), x))
