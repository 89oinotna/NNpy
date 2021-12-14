import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np


def read_monk(filename):

    dir_path = os.path.dirname(os.path.realpath(__file__))

    monk_dataset = pd.read_csv(dir_path+"/datasets/"+str(filename), sep=" ", names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id'])
    monk_dataset.set_index('id', inplace=True)
    monk_dataset = monk_dataset.sample(frac=1)
    labels = monk_dataset.pop('class').to_frame().values

    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.float32)
    return monk_dataset, labels


def read_cup(training=True, test=False, frac_train=1):

    columns_tr_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'label_x', 'label_y']
    columns_ts_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']

    dir_path = os.path.dirname(os.path.realpath(__file__))

    training_path = dir_path + "/datasets/ML-CUP21-TR.csv"
    test_path = dir_path + "/datasets/ML-CUP21-TS.csv"

    if training and not test:
        cup_tr_dataset = pd.read_csv(training_path, sep=",", names=columns_tr_names, skiprows=7)
        cup_tr_dataset = cup_tr_dataset.sample(frac=1)
        n_rows = cup_tr_dataset.shape[0]
        train_data = cup_tr_dataset.head(round(n_rows*frac_train))
        test_data = cup_tr_dataset.tail(n_rows-train_data.shape[0])
        train_labels = train_data.pop(['label_x', 'label_y'])
        test_labels = test_data.pop(['label_x', 'label_y'])
        return train_data, train_labels, test_data, test_labels
    if test and not training:
        cup_ts_dataset = pd.read_csv(test_path, sep=",", names=columns_ts_names, skiprows=7)
        cup_ts_dataset = cup_ts_dataset.sample(frac=1)
        return cup_ts_dataset
    if training and test:
        cup_tr_dataset = pd.read_csv(training_path, sep=",", names=columns_tr_names, skiprows=7)
        cup_tr_dataset = cup_tr_dataset.sample(frac=1)
        labels = cup_tr_dataset.pop(['label_x', 'label_y'])
        cup_ts_dataset = pd.read_csv(test_path, sep=",", names=columns_ts_names, skiprows=7)
        return cup_tr_dataset, labels, cup_ts_dataset
    else:
        return


if __name__ == "__main__":
    print(read_monk('monks-1.test')[1])