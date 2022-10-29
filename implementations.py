from helpers import *


def sigmoid(x):
    """
    Computes the sigmoid (logistic function) of a real-valued scalar
    Args:
        x: numpy array of shape=(N, )
    Returns:
        sgmd(x): numpy array of shape=(N, ). ith entry = sigmoid(x[i])
    """
    return 1 / (1 + np.exp(-x))


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
    prediction = sigmoid(np.dot(tx, w))
    # compute the log loss !
    loss_vector = y * np.log(prediction) + (np.ones(N) - y) * np.log(
        np.ones(N) - prediction
    )
    loss = -1 / (N**2) * np.sum(loss_vector) + lambda_ * (np.linalg.norm(w, 2) ** 2)
    return loss
    # N = len(y)
    # pred = sigmoid(tx.dot(w))
    # loss = 1/2*(np.ones(N)+y).T.dot(np.log(pred)) + 1/2*(np.ones(N)-y).T.dot(np.log(1 - pred)) + lambda_*(np.linalg.norm(w, 2)**2)
    # return np.squeeze(-loss)


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
    prediction = sigmoid(np.dot(tx, w))
    # compute the gradient of the log loss
    grad_log_loss = 1 / N * np.dot(tx.T, prediction - y)
    # add the gradient of the regularization term
    grad = grad_log_loss + 2 * lambda_ * w
    return grad


def compute_MSE(y, tx, w):
    """Calculate the loss using MSE
    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        w: shape=(M,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    w = w.reshape((len(w), 1))
    y = np.reshape(y, (len(y), 1))
    e = np.square(y - np.matmul(tx, w))
    N = len(y)
    loss = (1 / (2 * N)) * np.sum(e, axis=0)
    return loss[0]


# Linear Regression using Gradient Descent
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

    def compute_gradient(y, tx, w):
        N = y.shape[0]
        e = y - np.dot(tx, w)
        return -1 / N * np.dot(tx.transpose(), e)

    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = compute_MSE(y, tx, w)
    return w, loss

# Linear Regression using Stochastic Gradient Descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size = 1):
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

    def compute_stoch_gradient(y, tx, w):
        N = y.shape[0]
        e = y - np.dot(tx, w)
        return -1 / N * np.dot(tx.transpose(), e)

    w = initial_w
    for n_iter in range(max_iters):
        # Here we assume num_batches is 1; otherwise we need to update n_iter as well
        # This can be optimized since we shuffle everything each time we want one batch
        for curY, curTX in batch_iter(y, tx, batch_size):
            gradient = compute_stoch_gradient(curY, curTX, w)
            w = w - gamma * gradient
            loss = compute_MSE(y, tx, w)

    return w, loss

# Least Squares Regression using Normal Equations
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

    XtX = np.dot(tx.T, tx)
    w = np.dot(np.dot(np.linalg.inv(XtX), tx.T), y)
    loss = compute_MSE(y, tx, w)

    return w, loss

# Ridge Regression using Normal Equations
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

    N, D = tx.shape
    XtX = np.dot(tx.T, tx)
    XtX_Lambda = XtX + 2 * N * lambda_ * np.identity(D)
    w = np.dot(np.dot(np.linalg.inv(XtX_Lambda), tx.T), y)
    loss = compute_MSE(y, tx, w)
    return w, loss



def logistic_regression(y, tx, initial_w, max_iters, gamma):
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
    batch_size = 1
    for n_iter in range(max_iters):
        # # choose batch_size data points
        # data_points = np.random.randint(0, N, size=batch_size)
        # # pick out the datapoints
        # x_batch = tx[data_points]
        # y_batch = y[data_points]
        # # compute stochastic gradient
        gradient = compute_gradient_log_loss(y, tx, w)
        # update w by gradient
        w = w - (gamma * gradient)
    # compute log loss
    loss = compute_log_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
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
    batch_size = 1
    for n_iter in range(max_iters):
        # # choose batch_size data points
        # data_points = np.random.randint(0, N, size=batch_size)
        # # pick out the datapoints
        # x_batch = tx[data_points]
        # y_batch = y[data_points]
        # # compute stochastic gradient
        gradient = compute_gradient_log_loss(y, tx, w, lambda_)
        # update w by gradient
        w = w - (gamma * gradient)
    # compute log loss
    loss = compute_log_loss(y, tx, w)
    return w, loss

