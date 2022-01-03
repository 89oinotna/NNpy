import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os
import csv
import numpy as np


def read_monk(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    frac_train = 0.8
    monk_dataset = pd.read_csv(dir_path + "/datasets/monk/" + str(filename), sep=" ",
                               names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id'])
    monk_dataset.set_index('id', inplace=True)
    n_rows = monk_dataset.shape[0]
    train_data = monk_dataset.head(round(n_rows * frac_train))
    test_data = monk_dataset.tail(n_rows - train_data.shape[0])
    train_labels = train_data.pop('class').to_frame().values
    test_labels = test_data.pop('class').to_frame().values
    train_data = OneHotEncoder().fit_transform(train_data).toarray().astype(np.float32)
    test_data = OneHotEncoder().fit_transform(test_data).toarray().astype(np.float32)
    return train_data, train_labels, test_data, test_labels


def read_cup(frac_train=0.8):
    columns_tr_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'label_x', 'label_y']

    dir_path = os.path.dirname(os.path.realpath(__file__))

    training_path = dir_path + "/datasets/cup/ML-CUP21-TR.csv"

    cup_tr_dataset = pd.read_csv(training_path, sep=",", names=columns_tr_names, skiprows=7)
    cup_tr_dataset.pop('id')
    n_rows = cup_tr_dataset.shape[0]
    train_data = cup_tr_dataset.head(round(n_rows * frac_train))
    test_data = cup_tr_dataset.tail(n_rows - train_data.shape[0])
    train_labels = pd.DataFrame([train_data.pop(x) for x in ['label_x', 'label_y']]).transpose()
    test_labels = pd.DataFrame([test_data.pop(x) for x in ['label_x', 'label_y']]).transpose()
    """
    x = train_data.values  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(x)
    train_data = pd.DataFrame(data_scaled)
    """
    return train_data, train_labels.values, test_data, test_labels.values


def read_test_cup():
    columns_ts_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']

    dir_path = os.path.dirname(os.path.realpath(__file__))

    test_path = dir_path + "/datasets/cup/ML-CUP21-TS.csv"

    cup_ts_dataset = pd.read_csv(test_path, sep=",", names=columns_ts_names, skiprows=7)
    cup_ts_dataset = cup_ts_dataset.sample(frac=1)
    id = cup_ts_dataset.pop('id')
    return id, cup_ts_dataset


if __name__ == "__main__":
    print(read_monk('monks-1.test')[1])
