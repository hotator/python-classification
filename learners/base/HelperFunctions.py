#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import csv
import random


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    if len(numbers) - 1 == 0:
        return 1
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def dist(p1, p2):
    assert len(p1) == len(p2)
    temp = 0
    for i in range(len(p1)):
        temp += (p1[i] - p2[i])**2
    return math.sqrt(temp)


def get_nearest_index(value, vector):
    return min(range(len(vector)), key=lambda i: abs(vector[i]-value))


def get_most_common(lst):
    return max(set(lst), key=lst.count)


def load_csv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    res = []
    for i in range(len(dataset)):
        temp = [float(x) for x in dataset[i][:-1]]
        temp.append(dataset[i][-1].strip())
        res += [temp]
    return res


def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    training_set = []
    test_set = list(dataset)
    while len(training_set) < train_size:
        index = random.randrange(len(test_set))
        training_set.append(test_set.pop(index))
    return training_set, test_set
