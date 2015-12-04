#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" base class for classification learner """


class Learner(object):
    def __init__(self, trainset, testset):

        # attributes
        self.trainingset = trainset
        self.testset = testset
        self.separated = dict()

        self.predictions = []
        self.accuracy = 0

    def separate_by_class(self):
        for i in range(len(self.trainingset)):
            vector = self.trainingset[i]
            if vector[-1] not in self.separated:
                self.separated[vector[-1]] = []
            self.separated[vector[-1]].append(vector[:-1])

    def get_accuracy(self):
        correct = 0
        for i in range(len(self.testset)):
            if self.testset[i][-1] == self.predictions[i]:
                correct += 1
        self.accuracy = (correct/float(len(self.testset))) * 100.0

    def predict(self, input_vector):
        """
            predict something
            :param input_vector - new data
            :return class where vector should sort in
        """
        pass

    def get_predictions(self):
        for i in range(len(self.testset)):
            result = self.predict(self.testset[i])
            self.predictions.append(result)


