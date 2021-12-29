import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np


def read_monk(filename):

    dir_path = os.path.dirname(os.path.realpath(__file__))

    monk_dataset = pd.read_csv(dir_path+"/datasets/monk/"+str(filename), sep=" ", names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id'])
    monk_dataset.set_index('id', inplace=True)
    monk_dataset = monk_dataset.sample(frac=1)
    labels = monk_dataset.pop('class').to_frame().values

    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.float32)
    return monk_dataset, labels


def read_cup(training=True, test=False, frac_train=1):

    columns_tr_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'label_x', 'label_y']
    columns_ts_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']

    dir_path = os.path.dirname(os.path.realpath(__file__))

    training_path = dir_path + "/datasets/cup/ML-CUP21-TR.csv"
    test_path = dir_path + "/datasets/cup/ML-CUP21-TS.csv"

    if training and not test:
        cup_tr_dataset = pd.read_csv(training_path, sep=",", names=columns_tr_names, skiprows=7)
        cup_tr_dataset = cup_tr_dataset.sample(frac=1)
        cup_tr_dataset.pop('id')
        n_rows = cup_tr_dataset.shape[0]
        train_data = cup_tr_dataset.head(round(n_rows*frac_train))
        test_data = cup_tr_dataset.tail(n_rows-train_data.shape[0])
        train_labels = pd.DataFrame([train_data.pop(x) for x in ['label_x', 'label_y']]).transpose()
        test_labels = pd.DataFrame([test_data.pop(x) for x in ['label_x', 'label_y']]).transpose()
        return train_data, train_labels.values, test_data, test_labels.values
    if test and not training:
        cup_ts_dataset = pd.read_csv(test_path, sep=",", names=columns_ts_names, skiprows=7)
        cup_ts_dataset = cup_ts_dataset.sample(frac=1)
        cup_ts_dataset.pop('id')
        return cup_ts_dataset
    if training and test:
        cup_tr_dataset = pd.read_csv(training_path, sep=",", names=columns_tr_names, skiprows=7)
        cup_tr_dataset = cup_tr_dataset.sample(frac=1)
        cup_tr_dataset.pop('id')
        labels = pd.DataFrame([cup_tr_dataset.pop(x) for x in ['label_x', 'label_y']])
        cup_ts_dataset = pd.read_csv(test_path, sep=",", names=columns_ts_names, skiprows=7)
        cup_ts_dataset.pop('id')
        return cup_tr_dataset, labels.values, cup_ts_dataset
    else:
        return


if __name__ == "__main__":
    print(read_monk('monks-1.test')[1])