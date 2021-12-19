import numpy as np
import network as nn
# print(wi.xavier_init(2,2)[0:, 1:])
import activation_functions as af
import losses
import metrics
from input_reading import read_monk
import visualization as vis

monk_data, monk_label = read_monk('monks-1.test')
#print(monk_data[:1000, :])
optimizer = {
    'type_init': 'sgd',
  #  'nesterov':True,
    'ETA':0.0005,
    'momentum_window': 10,
    'ALPHA':0
}
wreg ={'type_init': 'tikonov', 'LAMBDA':0.001}
network = nn.NeuralNetwork.init([4, 1], 17, act_hidden='relu', act_out='sigmoid', w_init='monk', loss='mse',
                                metric='simple_class', optimizer=optimizer, weight_regularizer=wreg, epochs=400, minibatch_size=32)
metric, loss = network.fit(monk_data, monk_label)
vis.plot(loss, loss, metric, metric)
# x=np.array([[2,1],[2,1]])
# print(np.dot(np.array([2, 2]), x))
