"""

"""

import numpy as np

import caffe
from . import NeuralClassifier
from ..classification import SVMClassifier, ClassificationResult
from src import misc


class CaffeSVM(NeuralClassifier):
    """

    """

    def __init__(self, net, cnn_layer, record_features=False, oversample=False, snapshot=None):
        try:
            self.feature_extractor = net(snapshot) # net should be a class specializing CaffeNet
        except Exception as e:
            print 'Error while creating ConvNet: is net a class specializing CaffeNet? %s' % e.message

        self.threshold = -np.inf
        self.layer = tuple(cnn_layer)  # list are mutable so not hashable
        self.record_features = record_features
        self.oversample = oversample
        self.classifier = SVMClassifier()

    def finetune(self, train_set, validation_set):
        self.feature_extractor.finetune(train_set, validation_set)

    def train_classifier(self, train_set):
        self.feature_extractor._current_dataset = train_set.descriptor
        try:
            features = train_set.features[self.layer]
            if features is None:
                raise KeyError  # to be caught right below
        except KeyError:
            record = train_set.name+self.feature_extractor.name if self.record_features else None
            features = self.feature_extractor.process_all(train_set.as_dataframe()['filename'], cnn_layer=self.layer,
                                                          save=record)
            train_set.features[self.layer] = features

        self.classifier.train(features, train_set.as_dataframe()['label'])

        # from sklearn.linear_model import LogisticRegression
        # self.classifier = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        # self.classifier.fit(features, train_set.as_dataframe()['label'])

    def evaluate(self, test_set, evaluation_metric, save=None):
        self.feature_extractor._current_dataset = test_set.descriptor
        try:
            features = test_set.features[self.layer]
            if features is None:
                raise KeyError  # to be caught right below
        except KeyError:
            record = test_set.name+self.feature_extractor.name if self.record_features else None
            features = self.feature_extractor.process_all(test_set.as_dataframe()['filename'], cnn_layer=self.layer,
                                                          save=record)
            test_set.features[self.layer] = features

        try:
            labels = test_set.as_dataframe()['label']
            scores = self.classifier.classify(features, True)
            predictions = np.empty_like(labels)

            for i, score in enumerate(scores):  # TODO: refactor to numpy array style
                score = np.array(score)
                predictions[i] = ClassificationResult(score, self.threshold).get_first_class()[0]

            # predictions = self.classifier.classify(features, False)
            if save is not None:
                # escape filename
                filename = ''.join(c for c in 'results_'+save+'_'+str(self.layer) if c.isalnum() or c == '_')
                np.save(filename+'.npy', predictions)

            return evaluation_metric(predictions, labels)

            # errors = np.array(predictions != test_set.as_dataframe()['label'])
            # print zip(test_set.as_dataframe()['filename'][errors].values, predictions[errors])
            # print scores[errorsiric(predictions, labels)
        except AttributeError as e:
            print 'Error: %s. Have you trained the classifier?' % e.message

    def classify(self, image):
        try:
            # load image
            img = caffe.io.load_image(image)
        except IOError:
            raise ValueError("File not found: %s" % image)

        features = self.feature_extractor.process(img, self.layer, oversample=self.oversample)
        try:
            scores = self.classifier.classify(features, True)
            scores = scores.mean(0)

            from numpy import tanh
            scores = np.array([tanh(s) for s in scores])
            return ClassificationResult(scores, self.threshold)
        except AttributeError as e:
            print 'Error: %s. Have you trained the classifier?' % e.message
            raise Exception(e)

    def evaluate_threshold(self, test_set, evaluation_metric): # TODO: aggiungere nell'interfaccia
        self.feature_extractor._current_dataset = test_set.descriptor
        try:
            features = test_set.features[self.layer]
            if features is None:
                raise KeyError  # to be caught right below
        except KeyError:
            record = test_set.name+self.feature_extractor.name if self.record_features else None
            features = self.feature_extractor.process_all(test_set.as_dataframe()['filename'], cnn_layer=self.layer,
                                                          save=record)
            test_set.features[self.layer] = features

        try:
            labels = test_set.as_dataframe()['label']
            scores = self.classifier.classify(features, True)
            pred_probs = np.empty(labels.shape)

            for i, score in enumerate(scores):  # TODO: refactor to numpy array style
                # max score - 2nd score
                # score = np.array(score) - sorted(score, reverse=True)[1]
                # simply scores
                pred_probs[i] = np.array(score).max()

            return evaluation_metric(pred_probs, labels)

            # errors = np.array(predictions != test_set.as_dataframe()['label'])
            # print zip(test_set.as_dataframe()['filename'][errors].values, predictions[errors])
            # print scores[errorsiric(predictions, labels)
        except AttributeError as e:
            print 'Error: %s. Have you trained the classifier?' % e.message

class CaffeSVMFusion(NeuralClassifier):
    """
    Join features from given layers for the bounding box and the entire image
    """

    def __init__(self, net, net_bbox, cnn_layer, record_features=False, oversample=False, snapshot=None):
        """

        :param net:
        :param cnn_layer: a list of two layer names
        :param record_features:
        :param oversample:
        :param snapshot:
        :return:
        """
        try:
            # net and net_bbox should be classes specializing CaffeNet
            self.feature_extractor = net(snapshot)
            self.feature_extractor_bbox = net_bbox(snapshot)
        except Exception as e:
            print 'Error while creating ConvNet: is net a class specializing CaffeNet? %s' % e.message

        self.threshold = 0
        self.layer = cnn_layer
        self.record_features = record_features
        self.oversample = oversample
        self.classifier = SVMClassifier()

    def _extract_features(self, set):
        self.feature_extractor._current_dataset = set.descriptor
        try:
            features = set.features[self.layer]
            if features is None:
                raise KeyError  # to be caught right below
        except KeyError:
            record = set.name+self.feature_extractor.name if self.record_features else None
            features = self.feature_extractor.process_all(set.as_dataframe()['filename'], cnn_layer=self.layer,
                                                          save=record)
            set.features[self.layer] = features
        finally:
            return features

    def _extract_features_bbox(self, set):
        self.feature_extractor_bbox._current_dataset = set.descriptor
        try:
            features = set.features[self.layer]
            if features is None:
                raise KeyError  # to be caught right below
        except KeyError:
            record = set.name+self.feature_extractor_bbox.name if self.record_features else None
            features = self.feature_extractor_bbox.process_all(set.as_dataframe()['filename'], cnn_layer=self.layer,
                                                          save=record)
            set.features[self.layer] = features
        finally:
            return features

    def train_classifier(self, train_set):
        features = np.hstack(
            (self._extract_features(train_set), self._extract_features_bbox(train_set)))

        self.classifier.train(features, train_set.as_dataframe()['label'])


    def evaluate(self, test_set, evaluation_metric, save):
        features = np.hstack(
            (self._extract_features(test_set), self._extract_features_bbox(test_set)))

        try:
            labels = test_set.as_dataframe()['label']
            scores = self.classifier.classify(features, True)
            predictions = np.empty_like(labels)

            for i, score in enumerate(scores):  # TODO: refactor to numpy array style
                score = np.array(score)
                predictions[i] = ClassificationResult(score, self.threshold).get_first_class()[0]

            # predictions = self.classifier.classify(features, False)
            if save is not None:
                # escape filename
                filename = ''.join(c for c in 'results_'+save+'_'+self.layer if c.isalnum() or c == '_')
                np.save(filename+'.npy', predictions)

            return evaluation_metric(predictions, labels)

            # errors = np.array(predictions != test_set.as_dataframe()['label'])
            # print zip(test_set.as_dataframe()['filename'][errors].values, predictions[errors])
            # print scores[errorsiric(predictions, labels)
        except AttributeError as e:
            print 'Error: %s. Have you trained the classifier?' % e.message


    def classify(self, image):
        pass
