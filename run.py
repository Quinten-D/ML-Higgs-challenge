# -*- coding: utf-8 -*-
from utils import *
from implementations import *
import matplotlib as plt


def trainModel():
    """
    Trains the model using the training data
    Returns the trained weights, the MSE, the features that were not deemed useful, the mean of the kept features
    """

    y, features, ids, removed_features, means = load_training_data()
    tx = build_model_data(features)
    w, mse = least_squares(y, tx)
    return w, mse, removed_features, means


def runModel():
    """
    Trains the model and then run it on a test set to predict the results
    """

    w, mse_train, removed_features, means = trainModel()
    _, features, ids = load_test_data(removed_features, means)
    tx = build_model_data(features)
    y = np.dot(tx, w)
    y[y < 0] = -1
    y[y >= 0] = 1

    create_csv_submission(ids, y, "out.txt")


def plot_feature_output_relation():
    """
    Shows for each feature x potential relation between x and y
    If y seems to be random w.r.t. x, the feature is not useful
    """

    y, features, ids, removed_features = load_training_data()

    features = np.sort(features, axis=0)
    for i in range(20):
        plt.plt(features[:, i], y)
        plt.show()


def plot_feature_histogram():
    """
    Shows the most frequent values of each feature
    """

    y, features, ids, removed_features = load_training_data()
    x = np.array([j for j in range(features.shape[0])])

    features = np.sort(features, axis=0)
    for i in range(20):
        plt.hist(features[:, i])
        plt.show()


runModel()
