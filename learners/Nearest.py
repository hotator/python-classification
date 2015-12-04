#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" Nearest classification """

from .base.Learner import Learner
from .base.HelperFunctions import get_most_common, get_nearest_index


class Nearest(Learner):
    def __init__(self, trainset, testset):
        Learner.__init__(self, trainset, testset)

        # functions
        self.get_predictions()

    def predict(self, input_vector):
        class_vector = []
        set_of_attributes = zip(*self.trainingset)
        for i in range(len(input_vector[:-1])):
            n_index = get_nearest_index(input_vector[i], set_of_attributes[i])
            class_vector += [self.trainingset[n_index][-1]]

        return get_most_common(class_vector)
