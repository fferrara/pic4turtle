"""
Collection of Discriminator classes.
"""

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from . import Discriminator
from ..config import conf

class ClassificationResult(object):
    """
    The result of a classification.
    """
    def __init__(self, scores, threshold=0):
        self.scores = scores
        self.threshold = threshold

    def __is_enough_confidence(self):
        """
        Checks if the scores are high enough to provide an output class.
        :return: the value indicating if the confidence score is enough for an output
        """
        if self.scores.max() > self.threshold:
            return True

    def get_first_class(self):
        if self.__is_enough_confidence():
            return self.scores.argmax(), self.scores.max()
        else:
            # the "unclassified" class with confidence 0
            return conf.Unclassified, 0

    def get_all_scores(self):
        order = (-self.scores).argsort()
        return zip(order, self.scores[order])

class SVMClassifier(Discriminator):
    def __init__(self, C=1.0):
        self.normalizer = Normalizer(copy=True)
        self.stdscaler = StandardScaler(copy=True)

        self.classifier = LinearSVC(C=C, class_weight='auto')

    def train(self, features, labels):
        """

        :param features: numpy array with examples on rows and features on columns
        :param labels: numpy array or pandas dataframe with labels on rows
        """
        features = self.stdscaler.fit_transform(features)
        features = self.normalizer.fit_transform(features)

        self.classifier.fit(features, labels)

    def classify(self, features, return_scores):
        """

        :param features: numpy array with examples on rows and features on columns
        """
        features = self.stdscaler.transform(features)
        features = self.normalizer.transform(features)

        return self.classifier.decision_function(features) if return_scores else self.classifier.predict(features)
