#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" Meta classification """

from .base.Learner import Learner
from .base.HelperFunctions import get_most_common
from Nearest import Nearest
from Simple import Simple
from BlackHole import BlackHole


class Meta(Learner):
    def __init__(self, trainset, testset):
        Learner.__init__(self, trainset, testset)

        # TODO: vote_classes as parameter
        # voting classes
        vote_classes = [Nearest, Simple, BlackHole]
        self.vote_list = []
        for c in vote_classes:
            self.vote_list += [c(trainset, testset).predictions]

        # functions
        self.get_predictions()

    def get_predictions(self):
        for i in range(len(self.testset)):
            result = get_most_common(zip(*self.vote_list)[i])
            self.predictions.append(result)
