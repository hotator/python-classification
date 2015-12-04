#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" Simple classification """

from .base.Learner import Learner
from .base.HelperFunctions import dist


class Simple(Learner):
    def __init__(self, trainset, testset):
        Learner.__init__(self, trainset, testset)

        # functions
        self.get_predictions()

    def predict(self, input_vector):
        dist_vector = [dist(input_vector[:-1], p[:-1]) for p in self.trainingset]
        return self.trainingset[dist_vector.index(min(dist_vector))][-1]
