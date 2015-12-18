#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" multiple classification algorithms """

from prettytable import PrettyTable
from learners.base.HelperFunctions import load_csv, split_dataset
from learners import *
import numpy as np


def table_print(class_names, data_dict):
    t = PrettyTable(['Dataset'] + class_names)
    t.align['Dataset'] = 'l'
    for algo_name, algo_results in data_dict.iteritems():
        t.add_row([algo_name] + algo_results)
    print(t)


def calculate(data_name_list, class_list):
    res = dict()

    for dataset in data_name_list:
        data = load_csv('data/' + dataset + '.dat')
        train, test = split_dataset(data, 0.05)
        res[dataset] = []

        for c in class_list:
            algo = c(train, test)
            algo.get_accuracy()
            res[dataset] += [round(algo.accuracy, 2)]
    return res

if __name__ == '__main__':

    # get data
    # TODO: non numerical values like in car.dat
    #data = load_csv('challenge/points_bloed.csv')
    #data = load_csv('classdata/pima.dat')
    #train, test = split_dataset(data, 0.05)

    #classes = [NaivBayes, Nearest, Simple, Stupid, Stupid2, Random, BlackHole, Meta]
    classes = [Ensamble, Nearest]
    dataset_names = ['balance', 'banana', 'pima']
    #dataset_names = ['phoneme', 'haberman', 'contraceptive']
    #dataset_names = ['tae', 'titanic', 'hayes-roth']
    #dataset_names = ['bupa', 'newthyroid', 'monk-2']
    #dataset_names = ['appendicitis', 'glass', 'led7digit']

    #dataset_names = ['iris', 'wine', 'seeds', 'ecoli']
    #dataset_names += ['digits', 'yeast']
    #dataset_names = ['yeast']

    res_dict = calculate(dataset_names, classes)
    table_print([cls.__name__ for cls in classes], res_dict)

    """
    print('-------------------------')

    classes = [NaivBayes, Nearest, Simple, Stupid, Stupid2, Random, BlackHole, Meta]
    for c in classes:
        print(c.__name__)
        algo = c(train, test)
        algo.get_accuracy()
        print('-------------------------')
    """

    """
    # TODO: multi processing
    erg = []
    for _ in xrange(100):
        train, test = split_dataset(data, 0.05)
        naiv = NaivBayes(train, test)
        naiv.get_accuracy()

        #test = BlackHole(train, test)
        #test = Nearest(train, test)
        test = Simple(train, test)
        #test = Meta(train, test)

        test.get_accuracy()

        if naiv.accuracy <= test.accuracy:
            erg += [1]
        else:
            erg += [0]

    print(float(sum(erg)) / len(erg))
    """
