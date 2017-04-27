__author__ = 'flavio'
"""
This file contains the definition of interfaces, or behavior, of the neural classifier
that performs photo classification.
The classes implementing these interfaces can use different frameworks or model architectures as long as they respect
the methods definition.
The tracking of resources (model definition files, pre-trained weights) will be handled by the specific subclasses.
"""


class Discriminator(object):
    """
    A class discriminator based on pre-computed features.
    """

    def train(self, features, labels):
        raise NotImplementedError

    def classify(self, features, return_scores):
        raise NotImplementedError

from discrimination import SVMClassifier, ClassificationResult