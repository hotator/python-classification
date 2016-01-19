#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" Ensamble classification """

from itertools import combinations
import numpy as np
import random


def nearest_ensemble_modell(data, label): # "classification modell"
    label = [[a] for a in label]
    return np.concatenate((data, label), axis=1)


def nearest_ensemble(modelldata, datapoints, arg1):
    label = []
    # no argument greater than attributes of dataset or negative =0
    if arg1 > len(datapoints) or arg1 <= 0:
        arg1 = len(datapoints)
    for datapoint in datapoints:
        label += [int(predict(modelldata, datapoint, arg1))]
    return label


def predict(modelldata, input_vector, arg1):
    """
        :param
            modelldata - raw data
            input_vector - datapoint to classify
            arg1 - parameter between 1 and count of dimensions
        :return
            get most common class -> label
    """
    class_vector = []
    for val in get_combinations(range(len(input_vector)), arg1):
        dist_vector = []
        for train in modelldata:
            dist_vector += [(mydist([input_vector[i] for i in val], [train[i] for i in val]),
                             train[-1])]  # append distances from datapoint(combinations) and trainset
        class_vector += [min(dist_vector)[1]]

    return get_most_common(class_vector)


def get_combinations(vector, zahl):
    return list(combinations(vector, zahl))


def get_most_common(lst):
    return max(set(lst), key=lst.count)


def mydist(p1, p2):  # euclidean distance
    return np.linalg.norm(np.array(p1) - np.array(p2))


if __name__ == '__main__':

    import sklearn.datasets as skd
    data, label = skd.make_blobs(n_samples=150, n_features=4)

    train = data[:12]
    train_label = label[:12]
    test = data[12:]
    test_label = label[12:]

    modell = nearest_ensemble_modell(train, train_label)
    print(nearest_ensemble(modell, test, 1))
    """
    ###################################################################
    import sklearn.datasets as skd
    data, label = skd.make_blobs(n_samples=150, n_features=4)

    k = 1

    train = data[:12]
    train_label = label[:12]
    test = data[12:]
    test_label = label[12:]

    print train.shape
    print train_label.shape
    print test.shape
    print test_label.shape

    modell = nearest_ensemble_modell(train, train_label)
    prediction = predict(modell, test, k)
    """
