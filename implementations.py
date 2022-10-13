import numpy as np


#######################
### BASIC FUNCTIONS ###
#######################
def compute_MSE_loss(y, tx, w):
    """Calculate the loss using MSE

    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        w: shape=(M,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    w = w.reshape((len(w),1))
    y = np.reshape(y, (len(y), 1))
    e = (y - np.dot(tx, w))**2
    N = len(y)
    loss = (1/2*N) * np.sum(e, axis=0)
    return loss[0]

def compute_gradient(y, tx, w):
    """Computes the gradient at w for linear MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        w: shape=(M, ). The vector of model parameters.

    Returns:
        An array of shape (M, ) (same shape as w), containing the gradient of the loss at w.
    """
    # bring the parameters into the correct shape
    y = y.reshape((len(y), 1))
    w = w.reshape((len(w), 1))
    # compute gradient
    N = len(y)
    e = y-np.matmul(tx, w)
    gradient = -1/N*np.matmul(np.transpose(tx), e)
    return gradient.reshape((1,len(gradient)))[0]

def sigmoid(x):
    """
    Computes the sigmoid (logistic function) of a real-valued scalar
    Args:
        x: numpy array of shape=(N, )

    Returns:
        sgmd(x): numpy array of shape=(N, ). ith entry = sigmoid(x[i])
    """
    return 1/(1+np.exp(-x))

def compute_log_loss(y, tx, w, lambda_=0):
    """Calculate the log loss + L2 norm regularization term

    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        w: shape=(M,). The vector of model parameters.
        lambda_: hyperparameter for regularization

    Returns:
        loss: the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = len(y)
    # compute the predictions vector (shape (N,)) with probabilities P(y=1|x)
    prediction = sigmoid(np.ravel(np.dot(tx, w.reshape((len(w),1)))))
    # compute the log loss
    loss_vector = -y*np.log(prediction) - (np.ones(N)-y)*np.log(np.ones(N)-prediction)
    loss = 1/N * np.sum(loss_vector) + lambda_*np.norm(w, 2)**2
    return loss

def compute_gradient_log_loss(y, tx, w, lambda_=0):
    """Computes the gradient at w for a linear model with log loss cost function and L2 regularization.

    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        w: shape=(M, ). The vector of model parameters.
        lambda_: hyperparameter for regularization

    Returns:
        An array of shape (M, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = len(y)
    # compute the predictions vector (shape (N,)) with probabilities P(y=1|x)
    prediction = sigmoid(np.ravel(np.dot(tx, w.reshape((len(w),1)))))
    # compute the gradient of the log loss
    grad_log_loss = 1/N * np.dot(tx.T, prediction-y)
    # add the gradient of the regularization term
    grad = grad_log_loss + 2*lambda_*w
    return grad


##########################
### REQUIRED FUNCTIONS ###
##########################
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for a linear model using the MSE loss funtion.

    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        initial_w: shape=(M, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the model parameters for the last iteration of GD
        loss: the loss value (scalar) for the last iteration of GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        # update w by gradient
        w = w-(gamma*gradient)
    # compute mse loss
    loss = compute_MSE_loss(y, tx, w)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=32):
    """The Stochastic Gradient Descent algorithm (SGD) for a linear model using the MSE loss funtion.

    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        initial_w: shape=(M, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient

    Returns:
        w: the model parameters for the last iteration of SGD
        loss: the loss value (scalar) of the last iteration of SGD
    """
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        # choose batch_size data points
        data_points = np.random.randint(0, N, size=batch_size)
        # pick out the datapoints
        x_batch = tx[data_points]
        y_batch = y[data_points]
        # compute stochastic gradient
        stochastic_gradient = compute_gradient(y_batch, x_batch, w)
        # update w by gradient
        w = w-(gamma*stochastic_gradient)
    # compute mse loss
    loss = compute_MSE_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution by solving the normal equations.
       returns optimal weights and mse.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, the mse loss of the model with weights w.
    """
    # normal equations: X^TX w = X^Ty
    # first compute weights
    txT_dot_tx = np.dot(np.transpose(tx), tx)
    txT_dot_y = np.dot(np.transpose(tx), y.reshape((len(y), 1)))
    w = np.linalg.lstsq(txT_dot_tx, txT_dot_y.ravel(), rcond=-1)[0]
    # now compute the mean square error
    loss = compute_MSE_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Closed-form solution of linear model with MSE loss and L2-norm regularizer

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, the mse loss of the model with weights w + lambda_ times the L2 norm of w squared.
    """
    # w_ridge = A^-1 tx^Ty (with A = tx^Ttx + 2N*lambda_*I)
    N = len(y)
    D = len(tx[0])
    A = np.matmul(np.transpose(tx), tx) + 2*N*lambda_*np.identity(D)
    A_inverse = np.linalg.inv(A)
    w_ridge = np.matmul(A_inverse, np.matmul(np.transpose(tx), y.reshape((N, 1))))
    w = w_ridge.reshape((1, D))[0]
    loss = compute_MSE_loss(y, tx, w) + lambda_*np.linalg.norm(w, 2)**2
    return w, loss

def logistic_regression(y, tx, initial w, max iters, gamma):
    """logistic regression using SGD

    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        initial_w: shape=(M, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the model parameters for the last iteration of GD
        loss: the loss value (scalar) for the last iteration of GD
    """
    w = initial_w
    N = len(y)
    batch_size = 32
    for n_iter in range(max_iters):
        # choose batch_size data points
        data_points = np.random.randint(0, N, size=batch_size)
        # pick out the datapoints
        x_batch = tx[data_points]
        y_batch = y[data_points]
        # compute stochastic gradient
        stochastic_gradient = compute_gradient_log_loss(y_batch, x_batch, w)
        # update w by gradient
        w = w-(gamma*stochastic_gradient)
    # compute log loss
    loss = compute_log_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_ , initial w, max iters, gamma):
    """logistic regression with regularization term using SGD

    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        lambda_: hyperparameter for the regularization term
        initial_w: shape=(M, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the model parameters for the last iteration of GD
        loss: the loss value (scalar) for the last iteration of GD
    """
    w = initial_w
    N = len(y)
    batch_size = 32
    for n_iter in range(max_iters):
        # choose batch_size data points
        data_points = np.random.randint(0, N, size=batch_size)
        # pick out the datapoints
        x_batch = tx[data_points]
        y_batch = y[data_points]
        # compute stochastic gradient
        stochastic_gradient = compute_gradient_log_loss(y_batch, x_batch, w, lambda_)
        # update w by gradient
        w = w-(gamma*stochastic_gradient)
    # compute log loss
    loss = compute_log_loss(y, tx, w, lambda_)
    return w, loss
