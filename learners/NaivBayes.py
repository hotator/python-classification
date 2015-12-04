#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" naiv bayes classification """

from .base.Learner import Learner
from .base.HelperFunctions import mean, stdev
import math


class NaivBayes(Learner):
    def __init__(self, trainset, testset):
        Learner.__init__(self, trainset, testset)

        # Attributes
        self.summaries = {}
        self.probabilities = {}

        # functions
        self.summarize_by_class()
        self.get_predictions()

    @staticmethod
    def summarize(dataset):
        summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
        return summaries

    def summarize_by_class(self):
        self.separate_by_class()
        for classValue, instances in self.separated.iteritems():
            self.summaries[classValue] = self.summarize(instances)

    @staticmethod
    def calculate_probability(x, mean, stdev):
        if stdev == 0:
            return 0
        exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    def calculate_class_probabilities(self, input_vector):
        for classValue, classSummaries in self.summaries.iteritems():
            self.probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = input_vector[i]
                self.probabilities[classValue] *= self.calculate_probability(x, mean, stdev)

    def predict(self, input_vector):
        self.calculate_class_probabilities(input_vector)
        best_label, best_prob = None, -1
        for classValue, probability in self.probabilities.iteritems():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = classValue
        return best_label
