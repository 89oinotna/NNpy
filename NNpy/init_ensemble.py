import ensembling as en
import network as nn
import input_reading

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
                                  loss='mse', metric='mee', optimizer=optimizer_1, minibatch_size=32, epochs=500)

optimizer_2 = {
    'type_init': 'sgd',
    'ETA': 0.0675,
    'ALPHA': 0.775,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 7.5e-05
    }
}
network_2 = nn.NeuralNetwork.init(layer_sizes=[10, 70, 30, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='xavier',
                                  loss='mse', metric='mee', optimizer=optimizer_2, minibatch_size=32, epochs=500)

optimizer_3 = {
    'type_init': 'sgd',
    'ETA': 0.1,
    'ALPHA': 0.8,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 0.0001
    }
}
network_3 = nn.NeuralNetwork.init(layer_sizes=[40, 40, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='xavier',
                                  loss='mse', metric='mee', optimizer=optimizer_3, minibatch_size=32, epochs=500)

optimizer_4 = {
    'type_init': 'sgd',
    'ETA': 0.06,
    'ALPHA': 0.8,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 1e-05
    }
}
network_4 = nn.NeuralNetwork.init(layer_sizes=[38, 36, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='he',
                                  loss='mse', metric='mee', optimizer=optimizer_4, minibatch_size=32, epochs=1500)

optimizer_5 = {
    'type_init': 'sgd',
    'ETA': 0.07,
    'ALPHA': 0.9,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 0.0001
    }
}
network_5 = nn.NeuralNetwork.init(layer_sizes=[38, 36, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='he',
                                  loss='mse', metric='mee', optimizer=optimizer_5, minibatch_size=32, epochs=1500)

optimizer_6 = {
    'type_init': 'sgd',
    'ETA': 0.07,
    'ALPHA': 0.8,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 0.0001
    }
}
network_6 = nn.NeuralNetwork.init(layer_sizes=[38, 39, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='he',
                                  loss='mse', metric='mee', optimizer=optimizer_6, minibatch_size=32, epochs=1500)

optimizer_7 = {
    'type_init': 'sgd',
    'ETA': 0.06,
    'ALPHA': 0.9,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 0.0001
    }
}
network_7 = nn.NeuralNetwork.init(layer_sizes=[45, 45, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='he',
                                  loss='mse', metric='mee', optimizer=optimizer_7, minibatch_size=32, epochs=1500)

optimizer_8 = {
    'type_init': 'sgd',
    'ETA': 0.11,
    'ALPHA': 0.75,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 1e-06
    }
}
network_8 = nn.NeuralNetwork.init(layer_sizes=[36, 36, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='xavier',
                                  loss='mse', metric='mee', optimizer=optimizer_8, minibatch_size=100, epochs=750)

optimizer_9 = {
    'type_init': 'sgd',
    'ETA': 0.11,
    'ALPHA': 0.78,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 7.5e-05
    }
}
network_9 = nn.NeuralNetwork.init(layer_sizes=[36, 36, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='xavier',
                                  loss='mse', metric='mee', optimizer=optimizer_9, minibatch_size=32, epochs=950)

optimizer_10 = {
    'type_init': 'sgd',
    'ETA': 0.11,
    'ALPHA': 0.75,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 1e-05
    }
}
network_10 = nn.NeuralNetwork.init(layer_sizes=[36, 36, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='xavier',
                                  loss='mse', metric='mee', optimizer=optimizer_10, minibatch_size=64, epochs=1350)

optimizer_11 = {
    'type_init': 'sgd',
    'ETA': 0.07,
    'ALPHA': 0.85,
    'weight_regularizer': {
        'type_init': 'tikonov',
        'LAMBDA': 1e-05
    }
}
network_11 = nn.NeuralNetwork.init(layer_sizes=[38, 36, 2], input_size=10, act_hidden='sigmoid', act_out='id', w_init='he',
                                  loss='mse', metric='mee', optimizer=optimizer_11, minibatch_size=32, epochs=1500)


train_data, train_labels, valid_data, valid_labels = input_reading.read_cup(frac_train=0.8)

ensemble = en.Bagging(sample_size=len(train_data), models=[network_1, network_2, network_3, network_4, network_5, network_6, network_7,
                                                network_8, network_9, network_10, network_11])
final_training_error, final_validation_error, final_training_accuracy, final_validation_accuracy = ensemble.fit(train_data, train_labels, valid_data, valid_labels)

