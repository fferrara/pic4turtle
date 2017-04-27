"""
Helper functions.
"""

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import config
import caffe
from src.data.dataset import Dataset

def plot_losses(train_state, smooth=100, ntest=1):
    """
    Change smooth according to the plot length
    :param train_state:
    :param smooth:
    :return:
    """
    state = np.load(train_state)
    train_loss = np.array(state['train_l'])
    test_loss = np.array(state['test_l'])
    accuracy = np.array(state['acc'])

    if ntest == 2:
        tl1 = test_loss[::2]
        tl2 = test_loss[1::2]
        test_loss = np.hstack((tl1, tl2))
        acc1 = accuracy[::2]
        acc2 = accuracy[1::2]
        accuracy = np.hstack((acc1, acc2))

    if np.any(train_loss == 0):  # clean train loss
        train_loss = train_loss[:np.where(train_loss == 0)[0][0]]

    if np.any(test_loss == 0):  # clean train loss
        test_loss = test_loss[:np.where(test_loss == 0)[0][0]]

    if np.any(accuracy == 0):  # clean train loss
        accuracy = accuracy[:np.where(accuracy == 0)[0][0]]

    if smooth > 0:
        train_loss = moving_average(train_loss, smooth)
        test_loss = moving_average(test_loss, smooth / 10)
        accuracy = moving_average(accuracy, smooth / 10)

    plt.subplot(311)
    plt.plot(train_loss)
    plt.subplot(312)
    plt.plot(test_loss)
    plt.subplot(313)
    plt.plot(accuracy)
    plt.show()

def moving_average(a, n):
    # accumulate sum
    avg = np.cumsum(a, dtype=float)
    avg[n:] = avg[n:] - avg[:-n]
    return avg[n-1:] / n


def visualize_features(features, dataframe, nClasses=2):
    """
    Plot a 2-d version of the dataset, by applying the t-SNE algorithm on the CNN features
    :param features:
    :param labels:
    :return:
    """
    from tsne import bh_sne

    labels = dataframe['label'].values
    features_divided = []
    imgs_of_class = []
    colors = ["red", "blue" , "green", "orange", "purple"]

    for i in range(nClasses):
        examples_of_class_i = labels == i
        if np.sum(examples_of_class_i) > 100: # arbitrary value to garantee low perplexity
            imgs_of_class.append(dataframe['filename'][examples_of_class_i])
            to_process = features[examples_of_class_i]
            features_divided.append(bh_sne(to_process))

    for examples, color_ in zip(features_divided, colors):
        plt.scatter(examples[:, 0], examples[:, 1], color=color_)

    for i, img in enumerate(imgs_of_class[0]):
        img_name = img.split('/')[-1]
        if img_name[0:2] != 'n0':  # mainly not imagenet
            plt.annotate(img_name, xy=(features_divided[0][i, 0], features_divided[0][i, 1]))

    for i, img in enumerate(imgs_of_class[1]):
        img_name = img.split('/')[-1]
        if img_name[0:2] != 'n0':  # mainly not imagenet
                plt.annotate(img_name, xy=(features_divided[1][i, 0], features_divided[1][i, 1]))

    plt.show()

def enhance_images(images, path, convert_lab=False):
    """
    Apply CLAHE filter to enhance contrast.
    :param path: the path where the new images will be saved
    :param images: pandas dataframe containing filenames
    :param convert_lab: convert the image to LAB space before applying CLAHE
        if True. If False apply CLAHE on R, G and B separately.
    :return:
    """
    import cv2

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for filename in images:
        img = cv2.imread(filename)

        if convert_lab:
            lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab_planes = np.array(cv2.split(lab_image))
            lab_planes[0, ...] = clahe.apply(lab_planes[0])
            lab_image = cv2.merge(lab_planes)
            new_img = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
        else:
            new_img = np.zeros_like(img)
            new_img[:, :, 0] = clahe.apply(img[:, :, 0])
            new_img[:, :, 1] = clahe.apply(img[:, :, 1])
            new_img[:, :, 2] = clahe.apply(img[:, :, 2])

        cv2.imwrite(os.path.join(path, os.path.basename(filename)), new_img)


def get_nearest_neighbour(image, net, features, filenames, n=1):
    """
    Return the filename of the "closest matches" (using Euclidean distance) of a new image
    among the training examples
    :param n: the number of neighbours to be returned
    :param net: NeuralClassifier for extracting the features
    :param image: the new image path
    :param features: ndarray containing features for the training set, examples on rows
    :param filenames: iterable containing list of training images filenames
    """
    from scipy.spatial.distance import cdist
    try:
        # load image
        img = caffe.io.load_image(image)
    except IOError:
        raise ValueError("File not found: %s" % image)

    img_feats = net.feature_extractor.process(img, net.layer, net.oversample)
    img_feats = np.atleast_2d(img_feats)
    distances = cdist(img_feats, features)

    return filenames[np.argsort(distances, None)[:n]]

def similar_image_analysis(dataset, net):
    corrects = 0
    # to iterate tuples, need to fetch integer column header
    filename_i = np.where(dataset.val.as_dataframe().columns == 'filename')[0]
    label_i = np.where(dataset.val.as_dataframe().columns == 'label')[0]

    for img in dataset.val.as_dataframe().itertuples():
        img = img[1:]  # throw off row number
        print 'Current file %s (%d)' % (img[filename_i], img[label_i])

        # find the 5 NNs
        train_filenames_ndarray = dataset.train.as_dataframe()[filename_i].values
        neighbours = get_nearest_neighbour(img[filename_i], net, dataset.train.features[net.layer], train_filenames_ndarray, n=5)
        print 'Neighbours', neighbours

        # remove the NNs from the training set
        new_train = dataset.train.remove_examples(neighbours)

        # train again
        net.train_classifier(new_train)

        # classify the current image and check result
        result = net.classify(img[filename_i]).get_first_class()[0]
        print 'Classified %d' % result
        if result == img[label_i]:
            corrects += 1
        print '\n'

    print corrects

def get_saliency_map(img, net, label):
    """

    :param img: ndarray with the image
    :param net: NeuralClassifier with force_backward: true for Data Layer
    :param label: ndarray with shape (1, NUM_CLASSES) and true class set to 1
    :raise ValueError:
    """

    net.feature_extractor.process(img, net.layer, oversample=net.oversample)
    net = net.feature_extractor.net # access the raw pycaffe Net object
    bw = net.backward(**{net.outputs[0]: label})
    bw = bw['data']

    saliency = bw.squeeze()
    # As the paper says, we need the the
    # maximum magnitude of w across all colour channels
    return np.amax(np.abs(saliency), axis=0)

def plot_saliency_map(image, net, label):
    import matplotlib.cm as cm
    import caffe

    try:
        # load image
        img = caffe.io.load_image(image)
    except IOError:
        raise ValueError("File not found: %s" % image)

    label_ = np.zeros((1, 5)) # TODO: generalize
    label_[0, label] = 1.
    saliency = get_saliency_map(img, net, label_)

    plt.subplot(1,2,1)
    plt.imshow(saliency, cmap=cm.Blues)
    plt.subplot(1, 2, 2)
    img = caffe.io.resize_image(img, (224, 224)) # TODO: generalize
    plt.imshow(img)
    plt.show()

def show_test_errors(predictions):
    """
    Show images for which predictions is different from ground truth label.
    :param images: list with image files
    :param predictions: list with predictions
    :param labels: list with ground truth labels
    """
    d = Dataset(os.path.join(config.conf.AppPath, config.conf.DataPath))
    images = d.test.as_dataframe()['filename']
    labels = d.test.as_dataframe()['label']

    if len(images) != len(predictions) != len(labels):
        raise ValueError('images must contains same number of examples than predictions and labels')

    to_show = images[(predictions >= 0) & (predictions != labels)]
    for img in to_show:
        plt.imshow(mpimg.imread(img))
        plt.show()

def show_not_recognized(predictions):
    """
    Show images for which the system has not produced a valid classification.
    :param images: list with image files
    :param predictions: list with predictions
    """
    d = Dataset(os.path.join(config.conf.AppPath, config.conf.DataPath))
    images = d.test.as_dataframe()['filename']
    labels = d.test.as_dataframe()['label']

    if len(images) != len(predictions):
        raise ValueError('images must contains same number of examples than predictions')

    to_show = images[predictions < 0]
    for img in to_show:
        plt.imshow(mpimg.imread(img))
        plt.show()

def show_recognized(predictions):
    """
    Show images for which the system has not produced a valid classification.
    :param images: list with image files
    :param predictions: list with predictions
    """
    d = Dataset(os.path.join(config.conf.AppPath, config.conf.DataPath))
    images = d.test.as_dataframe()['filename']
    labels = d.test.as_dataframe()['label']

    if len(images) != len(predictions):
        raise ValueError('images must contains same number of examples than predictions')

    to_show = images[predictions >= 0]
    for img in to_show:
        plt.imshow(mpimg.imread(img))
        plt.show()