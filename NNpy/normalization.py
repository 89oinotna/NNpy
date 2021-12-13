import numpy as np


def normalize(feature_list=[], target_list=[], den_features=None, den_targets=None):
    """used to normalize features and/or targets

    Args:
        feature_list (list, optional): features to be normalized. Defaults to [].
        target_list (list, optional): targets to be normalized. Defaults to [].
        den_features ((float64,float64), optional): (mean,variance) for feature normalization. Defaults to None.
        den_targets ((float64,float64): (mean,variance) for target normalization. Defaults to None.

    Returns:
        (nparray, nparray, (mean,variance)): [description]
    """

    features = np.array(feature_list)
    targets = np.array(target_list)

    if den_features == None:
        den_features = (np.mean(features), np.std(features))
    if den_targets == None:
        den_targets = (np.mean(targets), np.std(targets))

    if len(features) > 0:
        features = (features - den_features[0]) / den_features[1]
    if len(targets) > 0:
        targets = (targets - den_targets[0]) / den_targets[1]

    return np.array(features), np.array(targets), den_features, den_targets


def denormalize(dataset, den_tuple):
    dataset = np.array(dataset)
    return dataset * den_tuple[1] + den_tuple[0]

