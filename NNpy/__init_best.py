import multiprocessing

import network as nn
import input_reading
from sklearn.model_selection import train_test_split

optimizer_1 = {
    'type_init': 'sgd',
    'ETA': 0.0675,
    'ALPHA': 0.85,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 7.5e-05
    }
}
network_1 = nn.NeuralNetwork.init(layer_sizes=[10, 70, 30, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='xavier',
                                  loss='mse', metric='mee', optimizer=optimizer_1, epochs=500)

minimum = 10
train_data, train_labels, test_data, test_labels = input_reading.read_cup(frac_train=0.8)
#train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.2)

for i in range(100):
    (final_training_error, final_validation_error), (final_training_accuracy, final_validation_accuracy) = \
        network_1.fit(train_data, train_labels, test_data, test_labels, early_stopping=True)
    if (final_validation_accuracy[-1]) < minimum:
        network_1.save(name="model 1")
        minimum = final_validation_accuracy[-1]


print("minimo: ", minimum)
(final_training_error, final_validation_error), (final_training_accuracy, final_validation_accuracy) = \
    nn.NeuralNetwork.load(name="model 1").fit(train_data, train_labels, test_data, test_labels, early_stopping=True)
print("Final validation accuracy: ", final_validation_accuracy[-1])
