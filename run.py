# -*- coding: utf-8 -*-
from utils import *
from implementations import *
import matplotlib as plt


def trainModel():
    """
    Trains the model using the training data
    Returns the trained weights, the MSE and the features that were not deemed useful
    """

    y, features, ids, removed_features = load_training_data()
    tx = build_model_data(features)
    w, mse = least_squares(y, tx)
    return w, mse, removed_features


def runModel():
    """
    Trains the model and then run it on a test set to predict the results
    """

    w, mse_train, removed_features = trainModel()
    _, features, ids = load_test_data(removed_features)
    tx = build_model_data(features)
    y = np.dot(tx, w)
    y[y < 0] = -1
    y[y >= 0] = 1

    create_csv_submission(ids, y, "out.txt")


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
