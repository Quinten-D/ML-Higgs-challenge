# -*- coding: utf-8 -*-
import numpy as np


def load_data(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
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
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


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


def sigmoid(x):
    """
    Computes the sigmoid (logistic function) of a real-valued scalar

    Args:
        x: numpy array of shape=(N, )
    
    Returns:
        sgmd(x): numpy array of shape=(N, ). ith entry = sigmoid(x[i]) 
    """
    return 1/(1+np.exp(-x))


def compute_mse(y, tx, w, lambda_=0):
    """Calculate the mean squared error corresponding to parameters w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
        lambda_: regularization parameter, set to 0 by default

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = np.shape(y)[0]
    e = y - np.dot(tx, w)
    mse = np.dot(e, e)/(2*N) + lambda_*np.dot(w, w)

    return mse


def compute_log_loss(y, tx, w, lambda_=0):
    """Calculate the log loss corresponding to parameters w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
        lambda_: regularization parameter, set to 0 by default

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    res = lambda_*np.dot(w, w)
    for i in range(np.shape(y)[0]):
        sgmd = sigmoid(np.dot(w, tx[i]))
        res -= y[i]*np.log(sgmd) + (1-y[i])*np.log(1-sgmd)

    return res 


def compute_gradient_MSE(y, tx, w):
    """Computes the gradient of the MSE at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = np.shape(y)[0]
    e = y-np.dot(tx, w)
    gradient = -np.dot(tx.T, e)/N
    return gradient


def compute_gradient_logloss(y, tx, w, lambda_=0):
    """Computes the gradient of the log logistic loss at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # see report for formula derivation 
    return np.dot(tx.T, y - sigmoid(np.dot(tx, w))) + 2*lambda_*w
