#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the main wrapper for the photo identification component.
The PhotoHandler is the enter point for the entire class structure.
It can be called by using RPC paradigm or embed this class into an external application code.
"""

import os
import numpy as np
import cPickle

from .neural import VGGNet, CaffeSVM, GoogleNet
from .data import Dataset
from config import conf
from . import misc
from .classification.performance import compute_partial_accuracy, compute_accuracy, get_confusion_matrix, \
    compute_precision_recall
from src.data.dataset import DatasetSplit
from src.neural.pure_caffe import PureVGGClassifier


class PhotoHandler(object):
    def __init__(self):
        """
        At the creation, PhotoHandler needs to instanciate the current classifier. It is an object
        that implements recognize(image) method.
        As new experiments run, the current classifier should be substituted with the best version at the time.

        :return: PhotoHandler instance
        """
        from numpy import tanh

        LAYER = ['pool5/7x7_s1']
        self.currentClassifier = CaffeSVM(GoogleNet, cnn_layer=LAYER, snapshot='google_svm/28092015_iter4000.caffemodel')
        self.currentClassifier.threshold = tanh(0.2) # normalized threshold

        # import pretrained classifier
        svm_trained = os.path.join(conf.AppPath, conf.ModelPath, 'google_svm', 'svm_trained.pkl')
        scaler_trained = os.path.join(conf.AppPath, conf.ModelPath, 'google_svm', 'scaler_trained.pkl')
        with open(svm_trained) as f:
            self.currentClassifier.classifier.classifier = cPickle.load(f)
        with open(scaler_trained) as f:
            self.currentClassifier.classifier.stdscaler = cPickle.load(f)

    def recognize(self, image):
        """
        The recognize(image) method is responsible for photo classification. It calls the current best classifier
        and returns the recognized class (or species) of the photo.

        :param image: the image file to be classified. The path needs to be accessible for reading.
        :return: the result of the classification. It is an instance of ClassificationResult.
        """
        results = self.currentClassifier.classify(image)
        # process results
        return results


if __name__ == '__main__':
    conf.DataPath = 'data_recorte'
    d = Dataset(os.path.join(conf.AppPath, conf.DataPath))

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    LAYER = ['pool5/7x7_s1']

    # a = PureVGGClassifier(snapshot='models/google4turtle_iter_1000.caffemodel')
    # a.FEATEX_MODEL = 'google_svm/featex.prototxt'

    a = CaffeSVM(GoogleNet, cnn_layer=LAYER, snapshot='models/google4turtle_iter_1000.caffemodel')
    # a.finetune(d.train, d.val)

    # a = CaffeSVM(VGGNet, cnn_layer=LAYER)

    a.train_classifier(d.train)
    # print 'Base accuracy: %2.02f' % (a.evaluate(d.val, compute_accuracy))
    # d.train.set_features(np.load(os.path.join(conf.AppPath, conf.DataPath, 'features_trainvggnet_'+LAYER+'.npy')), LAYER)

    # accuracy_avg = cross_validate(a, d, compute_accuracy,10)
    # print 'FC6 Val: ', accuracy_avg


    # d.train.set_features(np.load(os.path.join(conf.AppPath, conf.DataPath, 'features_trainvggnet_'+LAYER+'.npy')), LAYER)
    #d.test.set_features(np.load(os.path.join(conf.AppPath, conf.DataPath, 'features_other_data_thresholdvggnet_'+LAYER+'.npy')), LAYER)
    # a.train_classifier(d.train)

    # #

    prova = DatasetSplit.create_from_dir('other_data/threshold')
    a.evaluate_threshold(prova, compute_precision_recall)
    # print a.evaluate(d.val, compute_accuracy)
    # print a.evaluate(d.train, compute_accuracy)
    # print a.evaluate(misc.create_private_testset('fotos_test'), compute_partial_accuracy)
    # print a.evaluate(misc.create_private_testset('fotos_test_cut'), compute_partial_accuracy)
    # for foto in sorted(os.listdir(os.path.join(conf.AppPath, 'fotos_test'))):
    #     filename = os.path.join(conf.AppPath, 'fotos_test', foto)
    #     if any(filename.endswith(ext) for ext in ['.jpg', '.JPG', '.jpeg']):
    #         result = h.recognize(filename).get_first_class()
    #         print filename, ': ', d.get_classes_by_id()[result[0]]
    #         print ''


    # cf = a.evaluate(d.test, get_confusion_matrix)
    # print cf
    # print a.evaluate(d.val, compute_accuracy)
    # print a.evaluate(d.test, compute_accuracy)


    #
    # t2 = DatasetSplit.create_from_dir('fotos_test')
    # t3 = DatasetSplit.create_from_dir('fotos_test/cut')
    # t4 = DatasetSplit.create_from_dir('data_recorte', 'test_clean.txt')
    # print a.evaluate(t2, get_confusion_matrix)
    # print a.evaluate(t3, get_confusion_matrix)
    # print a.evaluate(t4, compute_accuracy, 'finetuned_1000')

    # acc1 = a.evaluate(d.val, compute_accuracy )
    # acc2 = a.evaluate(t2, compute_accuracy)
    # acc3 = a.evaluate(t3, compute_accuracy)
    # acc4 = a.evaluate(t4, compute_accuracy)

    # print '%2.01f, %2.01f' % (acc2*100, acc3*100)


    # for foto in sorted(os.listdir(os.path.join(conf.AppPath, 'fotos_test'))):
    #     filename = os.path.join(conf.AppPath, 'fotos_test', foto)
    # #     result =  h.recognize(filename).get_first_class()
    #     if any(filename.endswith(ext) for ext in ['.jpg', '.JPG', '.jpeg']):
    #         result = a.classify(filename).get_first_class()
    #         print filename, ': ', d.get_classes_by_id()[result[0]]
    #         print ''
    # #
    #
    # accuracy_avg = cross_validate(a, d, compute_accuracy, 5)
    # print 'Accuracy: ', accuracy_avg

    # for foto in sorted(os.listdir(os.path.join(conf.AppPath, 'fotos_test'))):
    #     filename = os.path.join(conf.AppPath, 'fotos_test', foto)
    #     #     result =  h.recognize(filename).get_first_class()
    #     if any(filename.endswith(ext) for ext in ['.jpg', '.JPG', '.jpeg']):
    #         result = a.classify(filename).get_first_class()
    #         print filename, ': ', d.get_classes_by_id()[result[0]], ', ', result[1]
    # for foto in sorted(os.listdir(os.path.join(conf.AppPath, 'fotos_test/cut'))):
    #     filename = os.path.join(conf.AppPath, 'fotos_test/cut', foto)
    #     #     result =  h.recognize(filename).get_first_class()
    #     if any(filename.endswith(ext) for ext in ['.jpg', '.JPG', '.jpeg']):
    #         result = a.classify(filename).get_first_class()
    #         print filename, ': ', d.get_classes_by_id()[result[0]], ', ', result[1]


    # print 'Accracy: %.03f' % cross_validate(a, d, compute_accuracy, 5)


    # accuracy_avg = cross_validate(a, d, compute_accuracy, 10)
    # print 'FC7 Val: ', accuracy_avg
    # accuracy_avg = cross_validate(a, d, compute_accuracy, 10)
    # print 'Train: ', accuracy_avg
    #


    ######## GOOGLENET CODE
    # train_tot = np.hstack((np.load('features_traingooglenet_loss1fc.npy'), np.load(
    #     'features_traingooglenet_loss2fc.npy'), np.load('features_traingooglenet_pool57x7_s1.npy')))
    # d.train.set_features(train_tot, 'custom')
    #
    # a.feature_extractor._current_dataset = d.test.descriptor
    # test_loss1 = a.feature_extractor.process_all(d.test.as_dataframe()['filename'], 'loss1/fc')
    # test_loss2 = a.feature_extractor.process_all(d.test.as_dataframe()['filename'], 'loss2/fc')
    # test_pool5 = a.feature_extractor.process_all(d.test.as_dataframe()['filename'], 'pool5/7x7_s1')
    # test_tot = np.hstack((test_loss1, test_loss2, test_pool5))
    # d.test.set_features(test_tot, 'custom')
    #
    # f1 = misc.create_private_testset('fotos_test')
    # a.feature_extractor._current_dataset = f1.descriptor
    # f1_tot = np.hstack((a.feature_extractor.process_all(f1.as_dataframe()['filename'], 'loss1/fc'),
    #                     a.feature_extractor.process_all(f1.as_dataframe()['filename'], 'loss2/fc'),
    #                     a.feature_extractor.process_all(f1.as_dataframe()['filename'], 'pool5/7x7_s1')))
    # f1.set_features(f1_tot, 'custom')
    #
    # f2 = misc.create_private_testset('fotos_test_cut')
    # a.feature_extractor._current_dataset = f2.descriptor
    # f2_tot = np.hstack((a.feature_extractor.process_all(f2.as_dataframe()['filename'], 'loss1/fc'),
    #                     a.feature_extractor.process_all(f2.as_dataframe()['filename'], 'loss2/fc'),
    #                     a.feature_extractor.process_all(f2.as_dataframe()['filename'], 'pool5/7x7_s1')))
    # f2.set_features(f2_tot, 'custom')
    #######
