import os
import numpy as np
import sys

from google.protobuf import text_format

import caffe
from caffe.proto import caffe_pb2
from . import NeuralClassifier
from ..classification import ClassificationResult
from ..config import conf


class PureCaffeClassifier(NeuralClassifier):
    """

    """

    def __init__(self, snapshot=None):
        # Prototxt files of CaffeNet model ###
        # Model definition (useful for classification)
        self.DEPLOY_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                  'pure_caffe/deploy.prototxt')
        # Pretrained weights
        self.PRETRAINED = os.path.join(conf.AppPath, conf.ModelPath,
                                       'bvlc_reference_caffenet.caffemodel')
        # Net snapshot
        if snapshot is not None:
            self.SNAPSHOT = os.path.join(conf.AppPath, conf.ModelPath,
                                         snapshot)
        else:
            self.SNAPSHOT = self.PRETRAINED
        # ImageNet mean values
        self.MEAN = np.array([104, 117, 123])
        # Model definition for training and validation (with loss and labels)
        self.TRAIN_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                        'pure_caffe/train_val.prototxt')
        # SGD optimizer definition
        self.SOLVER = os.path.join(conf.AppPath, conf.ModelPath,
                                   'pure_caffe/solver.prototxt')

        if conf.UseGPU == 'true':
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        # os.environ['GLOG_minloglevel'] = '2' ????
        self.net = None
        self.threshold = 0

    def __set_data_paths(self, train, validation):
        # Code for set the right paths within the prototxt files
        net_config = caffe_pb2.NetParameter()
        with open(self.TRAIN_MODEL) as f:
            text_format.Merge(str(f.read()), net_config)

        solver_config = caffe_pb2.SolverParameter()
        with open(self.SOLVER) as f:
            text_format.Merge(str(f.read()), solver_config)

        # modify data structure
        net_config.layer[0].image_data_param.source = train
        net_config.layer[1].image_data_param.source = validation
        solver_config.net = self.TRAIN_MODEL

        # write new prototxt
        new_net_config = text_format.MessageToString(net_config)
        with open(self.TRAIN_MODEL, 'w') as f:
            f.write(new_net_config)
        new_solver_config = text_format.MessageToString(solver_config)
        with open(self.SOLVER, 'w') as f:
            f.write(new_solver_config)

    def finetune(self, train_set_path, validation_set_path):
        # Setting the right paths for training (finetuning)
        self.__set_data_paths(train_set_path, validation_set_path)

        # Get useful values from solver file
        solver_config = caffe_pb2.SolverParameter()
        with open(self.SOLVER) as f:
            text_format.Merge(str(f.read()), solver_config)

        max_iter = solver_config.max_iter
        test_iter = solver_config.test_iter  # 128 images on each batch/iteration
        test_interval = solver_config.test_interval

        # Create the actual solver
        solver = caffe.SGDSolver(self.SOLVER)
        solver.net.copy_from(self.PRETRAINED)

        train_loss = np.zeros(max_iter)
        test_loss = np.zeros(max_iter / test_interval)
        accuracies = np.zeros(max_iter / test_interval)
        test_i = 0

        try:
            for it in xrange(max_iter):
                solver.step(1)
                train_loss[it] = solver.net.blobs['loss'].data
                if it % 50 == 0:
                    print 'Iteration %d, Finetune loss=%f' % (it, train_loss[it])

                if it % test_interval == 0:  # test net
                    test_loss_it = 0
                    test_accuracy = 0
                    for j in xrange(test_iter):
                        solver.test_nets[0].forward()

                        test_loss_it += solver.test_nets[0].blobs['loss'].data
                        test_accuracy += solver.test_nets[0].blobs['accuracy'].data

                    test_loss[test_i] = test_loss_it / test_iter
                    accuracies[test_i] = test_accuracy / test_iter
                    print 'Iteration %d, Test loss=%f, Accuracy=%f' % (it, test_loss[test_i], accuracies[test_i])
                    test_i += 1

                    # save training stats
                    with open('train_state.npz', 'wb') as f:
                        np.savez(f, train_loss=train_loss, test_loss=test_loss, accuracy=accuracies)
        finally:
            with open('train_state.npz', 'wb') as f:
                np.savez(f, train_loss=train_loss, test_loss=test_loss, accuracy=accuracies)

    def train_classifier(self, train_set):
        """
        Not implemented, since the class does not contain a Discriminator.
        """
        raise NotImplementedError

    def evaluate(self, test_set_df, evaluation_metric, save=None):
        """
        Evaluates the model against a test set.
        It classifies each image in the test set. The predictions and actual labels are passed to the evaluation
        metric to compute the given metric.
        :param test_set_df: a pandas DataFrame containing the images and labels of the test set
        :param threshold: the minimum confidence value for actually classifying a photo
        :param evaluation_metric: the EvaluationMetric chosen to assess performance
        :return: the ClassifierPerformance object
        """
        predictions = np.empty(test_set_df.shape[0])
        N = predictions.shape[0]

        for i, image in enumerate(test_set_df.ix[:, 0]):  # images in first column
            predictions[i] = self.classify(image).get_first_class()[0]  # just class, without confidence
            if i % 10 == 0:
                print '# {}/{}'.format(i, N)

        return evaluation_metric(predictions, test_set_df['label'])

    def classify(self, image):
        try:
            # load image
            img = caffe.io.load_image(image)
        except IOError:
            raise ValueError("File not found: %s" % image)

        # instantiate Classifier object,wrapper for basic Net interface
        if self.net is None:
            self.net = caffe.Classifier(self.DEPLOY_MODEL, self.SNAPSHOT, mean=self.MEAN,
                                    channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

        # perform classification
        scores = self.net.predict([img])

        # threshold is zero if another value not set by hand
        return ClassificationResult(scores[0], self.threshold)

class PureVGGClassifier(NeuralClassifier):
    """
    Pure test class, not supposed to be used on production.
    """
    def __init__(self, snapshot=None, oversample=False):
        # Model definition for offline validation
        self.FEATEX_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                         'vgg_svm/featex.prototxt')
        # Pretrained weights
        self.PRETRAINED = os.path.join(conf.AppPath, conf.ModelPath,
                                       'VGG_ILSVRC_16_layers.caffemodel')
        # Model definition (useful for classification)
        self.DEPLOY_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                  'vgg_svm/deploy.prototxt')
        # Net snapshot
        if snapshot is not None:
            self.SNAPSHOT = os.path.join(conf.AppPath, conf.ModelPath,
                                         snapshot)
        else:
            self.SNAPSHOT = self.PRETRAINED
        # ImageNet mean values
        self.MEAN = np.array([104, 117, 123])

        self.oversample = oversample
        self.threshold = 0
        self.net = None

        if conf.UseGPU == 'true':
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

    def _init_net_dataset(self):
        net_config = caffe_pb2.NetParameter()
        with open(self.FEATEX_MODEL) as f:
            text_format.Merge(str(f.read()), net_config)

        self.batch_size = net_config.layer[1].image_data_param.batch_size
        net_config.layer[1].image_data_param.source = self._current_dataset  ## TODO: DA CORREGGERE
        # write new prototxt
        new_net_config = text_format.MessageToString(net_config)
        with open(self.FEATEX_MODEL, 'w') as f:
            f.write(new_net_config)

        self.net = None
        self.net = caffe.Net(self.FEATEX_MODEL, self.SNAPSHOT, caffe.TEST)

    def train_classifier(self, train_set):
        # this NeuralClassifier has not a Discriminator
        raise NotImplementedError

    def classify(self, image):
        try:
            # load image
            img = caffe.io.load_image(image)
        except IOError:
            raise ValueError("File not found: %s" % image)

        if self.net is None:
            self.net = caffe.Net(self.DEPLOY_MODEL, self.SNAPSHOT, caffe.TEST)
        # setup preprocessing
        in_ = self.net.inputs[0]
        # resizing automatically to size defined in prototxt
        transformer = caffe.io.Transformer(
            {in_: self.net.blobs[in_].data.shape})
        transformer.set_transpose(in_, (2, 0, 1))
        transformer.set_mean(in_, self.MEAN)
        transformer.set_raw_scale(in_, 255)
        transformer.set_channel_swap(in_, (2, 1, 0))

        h, w = self.net.blobs[in_].data.shape[-2:]
        if self.oversample:
            image = caffe.io.resize_image(img, (256, 256)) # TODO: ajust 256 256
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample([image], (h, w))

            caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)

            for ix, crop in enumerate(input_):
                caffe_in[ix] = transformer.preprocess(in_, crop)

            self.net.blobs[in_].reshape(10, 3, h, w)
            self.net.blobs[in_].data[...] = caffe_in
        else:
            input_ = transformer.preprocess(in_, img)
            self.net.blobs[in_].reshape(1, 3, h, w)
            self.net.blobs[in_].data[...] = input_

        # net forward
        self.net.forward()

        probs = self.net.blobs['prob'].data
        return ClassificationResult(probs.mean(0), self.threshold)

    def evaluate(self, test_set, evaluation_metric, save):
        # get the softmax output from the net

        # init net
        self._current_dataset = test_set.descriptor
        self._init_net_dataset()
        dataset = test_set.as_dataframe()['filename']

        num_examples = dataset.shape[0]
        probs = np.zeros((num_examples, 5)) # TODO: adjust 5
        iterations = int(round((num_examples + 0.) / self.batch_size)) + 1

        for i in xrange(iterations):
            if i % 10 == 0:
                sys.stdout.write('#')
            # net forward by batches
            self.net.forward()
            prs = self.net.blobs['prob'].data.copy()
            for j in xrange(self.batch_size):
                f_i = self.batch_size * i + j

                if f_i < num_examples:
                    probs[f_i] = prs[j].flatten()

        sys.stdout.write('\n')

        predictions = np.argmax(probs, 1)
        labels = test_set.as_dataframe()['label']

        return evaluation_metric(predictions, labels)

    def evaluate_threshold(self, test_set, evaluation_metric): # TODO: aggiungere nell'interfaccia
        self._current_dataset = test_set.descriptor
        self._init_net_dataset()

        dataset = test_set.as_dataframe()['filename']

        num_examples = dataset.shape[0]
        probs = np.zeros((num_examples, 5)) # TODO: adjust 5
        iterations = int(round((num_examples + 0.) / self.batch_size)) + 1

        for i in xrange(iterations):
            if i % 10 == 0:
                sys.stdout.write('#')
            # net forward by batches
            self.net.forward()
            prs = self.net.blobs['prob'].data.copy()
            for j in xrange(self.batch_size):
                f_i = self.batch_size * i + j

                if f_i < num_examples:
                    probs[f_i] = prs[j].flatten()

        sys.stdout.write('\n')

        try:
            labels = test_set.as_dataframe()['label']
            pred_probs = probs.max(axis=1)

            return evaluation_metric(pred_probs, labels)

            # errors = np.array(predictions != test_set.as_dataframe()['label'])
            # print zip(test_set.as_dataframe()['filename'][errors].values, predictions[errors])
            # print scores[errorsiric(predictions, labels)
        except AttributeError as e:
            print 'Error: %s. Have you trained the classifier?' % e.message

