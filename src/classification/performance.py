"""

"""

import numpy as np

def get_confusion_matrix(predictions, labels):
    """
    Computes and returns the confusion matrix for a list of predictions.
    Labels must contain values between 0 and C, the number of classes
    Predictions may contain values equal to -1, meaning that there is no prediction for that example. Such examples
    will not be included in the confusion matrix,

    Labels as rows, predictions as columns.
    :param predictions: numpy array with predictions
    :param labels: numpy array with labels
    :param classes: the number of classes
    :return: a CxC matrix expressing the confusion matrix
    """
    if len(predictions) != len(labels):
        raise ValueError('predictions must contains same number of examples than labels')

    classes = max(np.max(labels), np.max(predictions)) + 1
    matrix = np.zeros((classes, classes))
    for (p, l) in zip(predictions, labels):
        if p >= 0:
            matrix[l, p] += 1

    return matrix


def compute_accuracy(predictions, labels):
    """
    Computes and returns the accuracy value for a list of predictions.
    Labels must contain values between 0 and C, the number of classes
    Predictions may contain values equal to -1, meaning that there is no prediction for that example. Such examples
    will not be included in the confusion matrix,

    :param predictions: numpy array with predictions
    :param labels:  numpy array with labels
    :return: a float value expressing accuracy value
    """

    if len(predictions) != len(labels):
        raise ValueError('predictions must contains same number of examples than labels')
    n = len(predictions)

    # retrieve only the predicted class > 0
    valid_predictions = predictions >= 0
    predictions = predictions[valid_predictions]
    labels = labels[valid_predictions]

    correct = np.sum(predictions == labels)
    try:
        return (correct + 0.) / np.sum(valid_predictions)
    except ZeroDivisionError:
        print 'Zero valid predictions'
        return 0.

def compute_partial_accuracy(predictions, labels):
    """
    Computes and returns the accuracy value for a list of predictions.
    Labels must contain values between 0 and C, the number of classes
    Predictions may contain values equal to -1, meaning that there is no prediction for that example. Such examples
    will not be included in the confusion matrix.
    This functions, differently from compute_accuracy, returns a dictionary containing accuracy for classified
    examples and percentage of unclassified examples.

    :param predictions: numpy array with predictions
    :param labels:  numpy array with labels
    :return: a dictionary with keys = ('classified', 'unclassified')
    """

    if len(predictions) != len(labels):
        raise ValueError('predictions must contains same number of examples than labels')
    n = len(predictions)

    # retrieve only the predicted class > 0
    valid_predictions = predictions >= 0
    predictions = predictions[valid_predictions]
    labels = labels[valid_predictions]

    correct = np.sum(predictions == labels)
    try:
        accuracy = {'classified': (correct + 0.) / np.sum(valid_predictions),
                    'unclassified': (np.sum(np.negative(valid_predictions)) + 0.) / n}
    except ZeroDivisionError:
        accuracy = {'classified': 0.,
                    'unclassified': 1.}

    return accuracy

def compute_precision_recall(pred_probs, labels):
    """
    Plot precision-recall pairs for different probability thresholds.
    :param pred_probs:
    :param labels:
    :return:
    """

    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, pred_probs)

    with open('pre_recall.npz', 'wb') as f:
        np.savez(f, precision=precision, recall=recall, thresholds=thresholds)
    return precision, recall, thresholds

def cross_validate(model, dataset, evaluation_fun, k=10):
    """
    Performs K-fold cross-validation.
    :param model:
    :param dataset:
    :param evaluation_fun:
    :param k:
    """
    avg_metric = None

    for i in range(k):
        dataset.create_datasets(0.8)
        model.train_classifier(dataset.train)
        metric = model.evaluate(dataset.test, evaluation_fun)
        print 'Iteration %d, metric %.02f' % (i+1, metric)
        if i == 0:
            avg_metric = metric
        else:
            avg_metric += metric

    return avg_metric / (k + 0.)