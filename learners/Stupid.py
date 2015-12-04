#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" Stupid classification """

from .base.Learner import Learner
from .base.HelperFunctions import get_most_common
import random


class Stupid(Learner):
    def __init__(self, trainset, testset):
        Learner.__init__(self, trainset, testset)

        # functions
        self.get_predictions()

    def predict(self, input_vector):
        return self.trainingset[0][-1]


class Stupid2(Learner):
    def __init__(self, trainset, testset):
        Learner.__init__(self, trainset, testset)

        self.value = get_most_common(zip(*trainset)[-1])

        # functions
        self.get_predictions()

    def predict(self, input_vector):
        return self.value


class Random(Learner):
    def __init__(self, trainset, testset):
        Learner.__init__(self, trainset, testset)

        self.possible_classes = list(set(zip(*trainset)[-1]))

        # functions
        self.get_predictions()

    def predict(self, input_vector):
        return random.choice(self.possible_classes)
