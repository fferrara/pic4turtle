__author__ = 'flavio'
"""
This file contains the definition of interfaces, or behavior, of the neural classifier
that performs photo classification.
The classes implementing these interfaces can use different frameworks or model architectures as long as they respect
the methods definition.
The tracking of resources (model definition files, pre-trained weights) will be handled by the specific subclasses.
"""

class NeuralEnsemble(object):
    """
    An Ensemble of trained NeuralClassifiers.
    """

    def add_to_ensemble(self, classifier):
        raise NotImplementedError

    def classify(self, image):
        raise NotImplementedError

class NeuralClassifier(object):
    """
    A single classifier using a Convolutional Neural Network for feature extraction.
    """

    def classify(self, image):
        """
        The method useful to classify a single image, once the classifier is trained. It takes an image path, which
        needs to be accessible, and returns an object containing the classification confidence for each possible class.

        :param image: the image path to be classified.
        :return: ClassificationResult object containing the output of the classification.
        """
        raise NotImplementedError

    def train_classifier(self, train_set):
        """
        Traines the classifier, or discriminator, in order to classify image features into classes.
        The classifier will fit the specified train_set.
        For this method to be called, the NeuralClassifier should contain an instance of Discriminator class.

        :param train_set: a DatasetSplit instance containing the training examples and labels
        """
        raise NotImplementedError

    def evaluate(self, test_set, evaluation_metric, save):
        """
        Evaluates a trained NeuralClassifier against a test set that it has never seen before. The evaluation is
        performed using the given evaluation metric. evaluation_metric object must implement correctly the compute()
        method.
        If a valid string is provided as save, the results of the classication will be stored in a file.
        :param save: the identifier of the dataset that will be included into the filename
        :param: test_set: the dataset containing the test examples
        :param evaluation_metric: a function computing the evaluation metric
        :return: the result of the evaluation produced by the evaluation_metric function
        """
        raise NotImplementedError

class FeatureExtractor(object):
    """
    A CNN-based feature extractor. It is used to compute CNN features from an image.
    """

    def process(self, image, cnn_layer):
        raise NotImplementedError

    def process_all(self, dataset, cnn_layer, save):
        raise NotImplementedError

    def finetune(self, train_set, validation_set):
        """
        Fine-tunes (or train from stratch) a FeatureExtractor using the given sets.
        The train set is used for the actual training, while the validation set is the one the model is
        evaluated against, during training. Both sets needs to have
        the same structure, that is, the same classes.
        Train set is usually way larger than validation set. The sets should not overlap.

        :param train_set: the dataset containing the training examples
        :param validation_set: the dataset containing the validation examples
        """
        raise NotImplementedError

from caffe_featex import CaffeNet, VGGNet, GoogleNet
from caffe_svm import CaffeSVM
from pure_caffe import PureCaffeClassifier, PureVGGClassifier