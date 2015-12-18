#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" Ensamble classification """

from .base.Learner import Learner
from .base.HelperFunctions import get_most_common, get_nearest_index, get_combinations, mydist


class Ensamble(Learner):
    def __init__(self, trainset, testset, num_vals=2):
        Learner.__init__(self, trainset, testset)

        # vals
        self.num = num_vals

        # functions
        self.get_predictions()

    def predict(self, input_vector):
        input_vector = input_vector[:-1]
        class_vector = []
        for val in get_combinations(range(len(input_vector)), self.num):
            dist_vector = []
            for train in self.trainingset:
                dist_vector += [(mydist([input_vector[i] for i in val], [train[i] for i in val]), train[-1])]
            class_vector += [min(dist_vector)[1]]

        return get_most_common(class_vector)
