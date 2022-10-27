
from helpers import *

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
    prediction = sigmoid(np.dot(tx, w))
    # compute the log loss
    loss_vector = y*np.log(prediction) + (np.ones(N)-y)*np.log(np.ones(N)-prediction)
    loss = -1/N * np.sum(loss_vector) + lambda_*(np.linalg.norm(w, 2)**2)
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
    prediction = sigmoid(np.dot(tx, w))
    # compute the gradient of the log loss
    grad_log_loss = 1/N * np.dot(tx.T, prediction-y)
    # add the gradient of the regularization term
    grad = grad_log_loss + 2*lambda_*w
    return grad

def compute_MSE(y, tx, w, _lambda = 0):
    N = y.shape[0]
    e = y - np.dot(tx, w)
    return 1 / (2 * N) * np.sum(e ** 2)

def gradient_descent_Charbel(y, tx, initial_w, max_iters, gamma):
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


def stochastic_gradient_descent_Charbel(y, tx, initial_w, max_iters, gamma, batch_size = 1):
    def compute_stoch_gradient(y, tx, w):
        N = y.shape[0]
        e = y - np.dot(tx, w)
        return -1 / N * np.dot(tx.transpose(), e)

    w = initial_w
    for n_iter in range(max_iters):
        # Here we assume num_batches is 1; otherwise we need to update n_iter as well
        # This can definitely be optimized since we shuffle everything each time we want one batch
        for curY, curTX in batch_iter(y, tx, batch_size):
            gradient = compute_stoch_gradient(curY, curTX, w)
            w = w - gamma * gradient
            loss = compute_MSE(y, tx, w)

    return w, loss

def least_squares_Charbel(y, tx):
    XtX = np.dot(tx.T, tx)
    w = np.dot(np.dot(np.linalg.inv(XtX), tx.T), y)
    MSE = compute_MSE(y, tx, w)

    return w, MSE

def ridge_regression_Charbel(y, tx, lambda_):
    N, D = tx.shape
    XtX = np.dot(tx.T, tx)
    XtX_Lambda = XtX + 2 * N * lambda_ * np.identity(D)
    w = np.dot(np.dot(np.linalg.inv(XtX_Lambda), tx.T), y)
    return w, compute_MSE(y, tx, w)

# Linear Regression using Gradient Descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    return gradient_descent_Charbel(y, tx, initial_w, max_iters, gamma)

# Linear Regression using Stochastic Gradient Descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size = 1):
    return stochastic_gradient_descent_Charbel(y, tx, initial_w, max_iters, gamma, batch_size)

# Least Squares Regression using Normal Equations
def least_squares(y, tx):
    return least_squares_Charbel(y, tx)

# Ridge Regression using Normal Equations
def ridge_regression(y, tx, lambda_):
    return ridge_regression_Charbel(y, tx, lambda_)

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

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
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

def trainModel():
    features, y, ids = load_training_data()
    tx = build_model_data(features)
    w, mse = least_squares(y, tx)
    return w, mse

def runModel():
    features, _, ids = load_test_data()
    tx = build_model_data(features)
    w, mse_train = trainModel()
    y = np.dot(tx, w)
    y[y < 0] = -1
    y[y >= 0] = 1

    print(y)
    print(w)

    create_submission(ids, y, "out.txt")

#runModel()



