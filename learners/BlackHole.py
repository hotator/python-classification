#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" Black hole classification """

from .base.Learner import Learner
from .base.HelperFunctions import dist, mean


class BlackHole(Learner):
    def __init__(self, trainset, testset):
        Learner.__init__(self, trainset, testset)

        # attributes
        self.summaries = {}

        # functions
        self.summarize_by_class()
        self.get_predictions()

    def summarize_by_class(self):
        self.separate_by_class()
        for classValue, instances in self.separated.iteritems():
            self.summaries[classValue] = self.summarize(instances)

    @staticmethod
    def summarize(dataset):
        summaries = [mean(attribute) for attribute in zip(*dataset)]
        return summaries

    def predict(self, input_vector):
        erg = []
        for classValue, instances in self.summaries.iteritems():
            erg += [[dist(instances, input_vector[:-1]), classValue]]
        return min(erg)[1]
