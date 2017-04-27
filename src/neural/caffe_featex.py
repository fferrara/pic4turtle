"""
CNN features extraction through Caffe framework.
"""

import sys
import os
import numpy as np

from google.protobuf import text_format

from caffe.proto import caffe_pb2
import caffe
from . import FeatureExtractor
from ..config import conf


class CaffeFeatureExtractor(FeatureExtractor):
    def __init__(self):
        """
        This class needs to be specialized for a single CNN architecture.
        So, attributes are set to None and should be filled in the child class.
        """
        # The convnet
        self.name = None
        self.net = None

        # Caffe prototxts
        self.DEPLOY_MODEL = None
        self.FEATEX_MODEL = None
        self.TRAIN_MODEL = None
        self.SOLVER = None

        # Caffe binary caffemodel
        self.PRETRAINED = None
        self.SNAPSHOT = None

        # Mean value for Imagenet
        self.MEAN = None

        # finetuning parameters
        self.MAX_ITER = None
        self.TEST_INTERVAL = None

        if conf.UseGPU == 'true':
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        # helper fields for finetuning
        self.finetune_loss = None
        self.test_loss = None
        self.test_accuracy = None


    def _init_net(self, mode):
        self.mode = mode # for future checks

        if mode == 'deploy':
            self.net = caffe.Net(self.DEPLOY_MODEL, self.SNAPSHOT, caffe.TEST)
            # setup preprocessing
            in_ = self.net.inputs[0]
            # resizing automatically to size defined in prototxt
            self.transformer = caffe.io.Transformer(
                {in_: self.net.blobs[in_].data.shape})
            self.transformer.set_transpose(in_, (2, 0, 1))
            self.transformer.set_mean(in_, self.MEAN)
            self.transformer.set_raw_scale(in_, 255)
            self.transformer.set_channel_swap(in_, (2, 1, 0))
        elif mode == 'dataset':
            net_config = caffe_pb2.NetParameter()
            with open(self.FEATEX_MODEL) as f:
                text_format.Merge(str(f.read()), net_config)

            self.batch_size = net_config.layer[1].image_data_param.batch_size
            net_config.layer[1].image_data_param.source = self._current_dataset  ## TODO: DA CORREGGERE
            # write new prototxt
            new_net_config = text_format.MessageToString(net_config)
            with open(self.FEATEX_MODEL, 'w') as f:
                f.write(new_net_config)

            self.net = caffe.Net(self.FEATEX_MODEL, self.SNAPSHOT, caffe.TEST)

    def _parse_solver(self):
        # Code for fetching the hyperparameters from prototxt file
        net_config = caffe_pb2.NetParameter()
        with open(self.TRAIN_MODEL) as f:
            text_format.Merge(str(f.read()), net_config)

        solver_config = caffe_pb2.SolverParameter()
        with open(self.SOLVER) as f:
            text_format.Merge(str(f.read()), solver_config)

        self.train_batch_size = int(net_config.layer[0].data_param.batch_size)
        self.test_batch_size = int(net_config.layer[1].data_param.batch_size)
        self.display = int(solver_config.display)
        self.max_iter = int(solver_config.max_iter)
        self.test_interval = int(solver_config.test_interval)
        self.test_iter = int(solver_config.test_iter[0]) # a test_iter for each test_net
        self.iter_size = int(solver_config.iter_size)

    def process(self, image, cnn_layer, oversample):
        """

        :param image: single numpy array
        :param cnn_layer: a list with the layer names from which extract features
        :param oversample: if True, 10 crops are obtained from the image and the result is averaged
        """

        if self.net is None or self.mode == 'dataset':
            self._init_net(mode='deploy')

        # PREPROCESS
        # Resize to standardize input dimensions.
        #   caffe.io.resize_image(image, image_dims)
        # Transpose
        #   in_.transpose((2,0,1))
        # Channel swap
        #   in_[(2,0,1), :, :]
        # Scale to [0,255] for Imagenet models
        #   in_ *= 255
        # Normalize
        #   in_ -= mean
        #
        in_ = self.net.inputs[0]
        h, w = self.net.blobs[in_].data.shape[-2:]

        if oversample:
            image = caffe.io.resize_image(image, (256, 256)) # TODO: ajust 256 256
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample([image], (h, w))

            caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)

            for ix, crop in enumerate(input_):
                caffe_in[ix] = self.transformer.preprocess(in_, crop)

            self.net.blobs[in_].reshape(10, 3, h, w)
            self.net.blobs[in_].data[...] = caffe_in
        else:
            input_ = self.transformer.preprocess(in_, image)
            self.net.blobs[in_].reshape(1, 3, h, w)
            self.net.blobs[in_].data[...] = input_

        # net forward
        self.net.forward()

        # return stacked features
        blobs = [self.net.blobs[l].data.copy().squeeze() for l in cnn_layer]
        return np.hstack(blobs)

    def process_all(self, dataset, cnn_layer, save):
        """
        Extracts features from an entire dataset.
        If save is not None, the extracted features will be stored in a npy file on disk. In this case,
        save should be a string that will be included into the filename.

        :param dataset: a pandas dataframe
        :param cnn_layer: a list with the layer names from which extract features
        :param save: the identifier of the dataset that will be included into the filename
        :return: the extracted features
        """
        self._init_net(mode='dataset')

        num_examples = dataset.shape[0]
        blobs = [self.net.blobs[l].data.copy().squeeze() for l in cnn_layer]

        features = np.zeros((num_examples, np.hstack(blobs).shape[1]))
        iterations = int(round((num_examples + 0.) / self.batch_size)) + 1

        for i in xrange(iterations):
            if i % 10 == 0:
                sys.stdout.write('#')
            # net forward by batches
            self.net.forward()
            blobs = [self.net.blobs[l].data.copy().squeeze() for l in cnn_layer]
            feats = np.hstack(blobs)
            for j in xrange(self.batch_size):
                f_i = self.batch_size * i + j

                if f_i < num_examples:
                    features[f_i] = feats[j].flatten()

        # return
        sys.stdout.write('\n')

        if save is not None:
            # escape filename
            filename = ''.join(c for c in 'features_%s_%s' % (save, ''.join(cnn_layer)) if c.isalnum() or c == '_')
            np.save(filename+'.npy', features)
        return features

    def finetune(self, train_set, validation_set):
        # Setting the right paths for training (finetuning)
        self._parse_solver()
        # Create the actual solver
        solver = caffe.NesterovSolver(self.SOLVER)
        solver.net.copy_from(self.SNAPSHOT)
        solver.test_nets[0].copy_from(self.SNAPSHOT)

        train_examples = train_set.as_dataframe().shape[0]

        train_loss = np.zeros(self.max_iter)
        test_loss = np.zeros(self.max_iter / self.test_interval)
        accuracies = np.zeros(self.max_iter / self.test_interval)
        test_i = 0
        epoch = 0

        if self.finetune_loss is None:
            self.finetune_loss = 'loss'
        if self.test_loss is None:
            self.test_loss = 'loss'
        if self.test_accuracy is None:
            self.test_accuracy = 'accuracy'

        try:
            for i in xrange(self.max_iter):
                if (i * self.iter_size * self.train_batch_size) >= (train_examples*epoch):
                    print 'Starting epoch %d' % epoch
                    epoch += 1

                if i % self.test_interval == 0:  # test net
                    test_loss_it = 0
                    test_accuracy = 0
                    for j in xrange(self.test_iter):
                        solver.test_nets[0].forward()

                        test_loss_it += solver.test_nets[0].blobs[self.test_loss].data
                        test_accuracy += solver.test_nets[0].blobs[self.test_accuracy].data

                    test_loss[test_i] = test_loss_it / self.test_iter
                    accuracies[test_i] = test_accuracy / self.test_iter
                    print 'Iteration %d, Test loss=%f, Accuracy=%f' % (i, test_loss[test_i], accuracies[test_i])
                    test_i += 1

                    # save training stats
                    with open('train_state.npz', 'wb') as f:
                        np.savez(f, train_l=train_loss, test_l=test_loss, acc=accuracies)

                # one SGD step
                solver.step(1)
                train_loss[i] = solver.net.blobs[self.finetune_loss].data
                if i % self.display == 0:
                    print 'Iteration %d, Finetune loss=%f' % (i, train_loss[i])
        finally:
            # save training stats
            with open('train_state.npz', 'wb') as f:
                np.savez(f, train_l=train_loss, test_l=test_loss, acc=accuracies)

class CaffeNet(CaffeFeatureExtractor):
    """

    """
    def __init__(self, snapshot):
        super(CaffeNet, self).__init__()

        # Prototxt files of CaffeNet model ###
        # Model definition (useful for classification)
        self.DEPLOY_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                         'caffe_svm/deploy.prototxt')
        # Model definition for offline validation
        self.FEATEX_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                         'caffe_svm/featex.prototxt')
        # Model definition for finetuning
        self.TRAIN_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                        'caffe_svm/train_val.prototxt')
        # Pretrained weights
        self.PRETRAINED = os.path.join(conf.AppPath, conf.ModelPath,
                                       'bvlc_reference_caffenet.caffemodel')
        # Net snapshot
        if snapshot is None:
            snapshot = 'bvlc_reference_caffenet.caffemodel'
        self.SNAPSHOT = os.path.join(conf.AppPath, conf.ModelPath,
                                         snapshot)

        # SGD optimizer definition
        self.SOLVER = os.path.join(conf.AppPath, conf.ModelPath,
                                   'caffe_svm/solver.prototxt')
        # ImageNet mean values
        self.MEAN = np.array([104, 117, 123])

        self.name = 'caffenet' + snapshot

class VGGNet(CaffeFeatureExtractor):
    """
    """
    def __init__(self, snapshot):
        super(VGGNet, self).__init__()

        # Prototxt files of CaffeNet model ###
        # Model definition (useful for classification)
        self.DEPLOY_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                  'vgg_svm/deploy.prototxt')
        # Model definition for offline validation
        self.FEATEX_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                         'vgg_svm/featex.prototxt')
        # Model definition for finetuning
        self.TRAIN_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                  'vgg_svm/train_val.prototxt')
        # Pretrained weights
        self.PRETRAINED = os.path.join(conf.AppPath, conf.ModelPath,
                                       'VGG_ILSVRC_16_layers.caffemodel')
        # SGD optimizer definition
        self.SOLVER = os.path.join(conf.AppPath, conf.ModelPath,
                                   'vgg_svm/solver.prototxt')
        # Net snapshot
        if snapshot is None:
            snapshot = 'VGG_ILSVRC_16_layers.caffemodel'
        self.SNAPSHOT = os.path.join(conf.AppPath, conf.ModelPath,
                                         snapshot)

        # ImageNet mean values
        self.MEAN = np.array([104, 117, 123])

        self.name = 'vggnet' + snapshot

class GoogleNet(CaffeFeatureExtractor):
    """
    """

    def __init__(self, snapshot):
        super(GoogleNet, self).__init__()

        # Prototxt files of CaffeNet model ###
        # Model definition (useful for classification)
        self.DEPLOY_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                         'google_svm/deploy.prototxt')
        # Model definition for offline validation
        self.FEATEX_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                         'google_svm/featex.prototxt')
        # Model definition for finetuning
        self.TRAIN_MODEL = os.path.join(conf.AppPath, conf.ModelPath,
                                        'google_svm/train_val.prototxt')
        # Pretrained weights
        self.PRETRAINED = os.path.join(conf.AppPath, conf.ModelPath,
                                       'bvlc_googlenet.caffemodel')
        # SGD optimizer definition
        self.SOLVER = os.path.join(conf.AppPath, conf.ModelPath,
                                   'google_svm/solver.prototxt')
        # Net snapshot
        if snapshot is None:
            snapshot = 'bvlc_googlenet.caffemodel'
        self.SNAPSHOT = os.path.join(conf.AppPath, conf.ModelPath,
                                         snapshot)

        # ImageNet mean values
        self.MEAN = np.array([104, 117, 123])

        self.name = 'googlenet' + snapshot
