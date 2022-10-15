# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import csv

def load_old_data(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "Data/height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5 / 0.454, 55.3 / 0.454]])

    return height, weight, gender

def build_old_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx



def load_data(path_dataset):
    ids = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0])

    output = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1],
        converters={1: lambda x: -1 if b"b" in x else 1})

    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[col for col in range(2, 32)])

    features = []
    for i in range(0, 30):
        curFeature = data[:, i]
        curFeature[curFeature == -999] = np.mean(curFeature[curFeature != -999])
        standardize(curFeature)
        features.append(curFeature)

    return np.array(features).T, np.array(output), np.array(ids)

def load_training_data():
    return load_data("Data/train.csv")

def load_test_data():
    return load_data("Data/test.csv")

def create_submission(ids, y_pred, outputFileName):
    with open(outputFileName, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x -= mean_x
    std_x = np.std(x)
    x /= std_x
    return x, mean_x, std_x


def build_model_data(features):
    num_samples = features.shape[0]
    tx = np.c_[np.ones(num_samples), features]
    return tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
