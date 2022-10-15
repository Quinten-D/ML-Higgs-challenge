# -*- coding: utf-8 -*-
from helper import *
import numpy as np

# implementation of the 6 functions listed in the project description

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: the value of the mean squared error (a scalar), corresponding to the input parameters w.
    """
    # set initial weight
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient_MSE(y, tx, w)

        # update weights by gradient
        w = w - gamma*gradient

    mse = compute_mse(y, tx, w)
    return (w, mse)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=128):
    """Linear regression using stochastic gradient descent
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: the value of the mean squared error (a scalar), corresponding to the input parameters w.
    """
    # set initial weight
    w = initial_w

    for n_iter in range(max_iters):
        # draw a batch at random 
        indices  = np.random.randint(0, np.shape(y)[0], batch_size)
        batch_tx = np.array([tx[i] for i in indices])
        batch_y  = np.array([y[j] for j in indices])
        
        # compute gradient over the batch elements
        sgd = compute_gradient_MSE(batch_y, batch_tx, w)

        # update weights by stochastic gradient
        w = w - gamma*sgd

    mse = compute_mse(y, tx, w)
    return (w, mse)


def least_squares(y, tx):
    """Closed-form solution of least squares regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # get number of samples N
    N   = np.shape(y)[0]
    # compute Gram matrix
    A   = np.dot(tx.T, tx)
    b   = np.dot(tx.T, y)
    
    # solve for optimal weights and compute loss
    w   = np.linalg.lstsq(A, b, rcond=None)[0]
    mse = compute_mse(y, tx, w, lambda_=0)
    return (w, mse)


def ridge_regression(y, tx, lambda_):
    """Closed-form solution of least squares regression using Ridge regularization.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: regularization parameter, set to 0 by default    

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse_ridge: Ridge-regularized MSE (a scalar) corresponding to the input parameters w.
    """
    # get number of samples N
    N = np.shape(y)[0]

    lambda_prime = 2*N*lambda_
    A = np.dot(tx.T, tx)+lambda_prime*np.identity(np.shape(tx)[1])
    b = np.dot(tx.T, y)

    # solve for optimal weights and compute loss
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    mse = compute_mse(y, tx, w, lambda_)
    return (w, mse)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=128):
    """Logistic regression using stochastic gradient descent and Ridge regularization
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        lambda_: regularization parameter (scalar)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        logloss: the value of the loss (a scalar), corresponding to the input parameters w.
    """

    # set initial weight
    w = initial_w

    for n_iter in range(max_iters):
        # draw batch at random 
        indices  = np.random.randint(0, np.shape(y)[0], batch_size)
        batch_tx = np.array([tx[i] for i in indices])
        batch_y  = np.array([y[j] for j in indices])
        
        # compute gradient over the batch elements
        sgd = compute_gradient_logloss(batch_y, batch_tx, w, lambda_)

        # update weights by stochastic gradient
        w = w - gamma*sgd

    logloss = compute_log_loss(y, tx, w, lambda_)
    return (w, logloss)


def logistic_regression(y, tx, initial_w, max_iters, gamma, batch_size = 128):
    """Logistic regression using stochastic gradient descent
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    # set lambda_ to 0 and use reg_logistic_regression
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma, batch_size)