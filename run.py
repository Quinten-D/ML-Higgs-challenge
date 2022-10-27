# -*- coding: utf-8 -*-
from helpers import *
from implementations import *
import matplotlib as plt

def trainModel():
    """
    implementations charbel
    """

    y, features, ids, removed_features = load_training_data()
    tx = build_model_data(features)
    w, mse = least_squares(y, tx)
    return w, mse, removed_features

def runModel():
    """
    implementations charbel
    """

    w, mse_train, removed_features = trainModel()
    _, features, ids = load_test_data(removed_features)
    tx = build_model_data(features)
    y = np.dot(tx, w)
    y[y < 0] = -1
    y[y >= 0] = 1

    create_csv_submission(ids, y, "out.txt")


#Shows the most frequent values of each feature
def plot_feature_histogram():
    y, features, ids, removed_features = load_training_data()
    x = np.array([j for j in range(features.shape[0])])

    features = np.sort(features, axis=0)
    for i in range(20):
        plt.hist(features[:, i])
        plt.show()

runModel()