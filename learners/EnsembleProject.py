#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" Ensamble classification """

from itertools import combinations
import numpy as np
import random


def nearest_ensemble_modell(data, label):
    return np.concatenate((data, label), axis=1)


def nearest_ensemble(modelldata, datapoints, arg1):  # list of data to classify = datapoints
    label = []
    if arg1 > len(datapoints) or arg1 <= 0: # no argument greater than attributes of dataset or negative =0
        arg1 = len(datapoints)
    for datapoint in datapoints:
        label += [predict(modelldata, datapoint, arg1)]
    return label


def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    training_set = []
    test_set = list(dataset)
    while len(training_set) < train_size:
        index = random.randrange(len(test_set))
        training_set.append(test_set.pop(index))
    # return dataset[:train_size], dataset[train_size:]
    return np.array(training_set), np.array(test_set)


def predict(modelldata, input_vector, arg1):
    class_vector = []
    for val in get_combinations(range(len(input_vector)), arg1):  # combinations of dataattributes
        dist_vector = []  # vectorlist for distances to trainingset
        for train in modelldata:
            dist_vector += [(mydist([input_vector[i] for i in val], [train[i] for i in val]),
                             train[-1])]  # append distances from datapoint(combinations) and trainset
        class_vector += [min(dist_vector)[1]]  # add classes of nearest

    return get_most_common(class_vector)  # get most common class -> label


def get_combinations(vector, zahl):  # combinations
    return list(combinations(vector, zahl))


def get_most_common(lst):
    return max(set(lst), key=lst.count)


def mydist(p1, p2):  # euclidean distance
    return np.linalg.norm(np.array(p1) - np.array(p2))


if __name__ == '__main__':
    data = np.genfromtxt('../data/iris.dat', delimiter=',')
    trainset, testset = split_dataset(data, 0.05)

    train = trainset[:, :-1]

    label = trainset[:, -1:]

    test = testset[:, :-1]

    modell = nearest_ensemble_modell(train, label)
    print(nearest_ensemble(trainset, test, 2))
