# -*- coding: utf-8 -*-
import numpy as np

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
