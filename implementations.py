
from helpers import *

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
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return gradient_descent_Charbel(y, tx, initial_w, max_iters, gamma)

# Linear Regression using Stochastic Gradient Descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size = 1):
    return stochastic_gradient_descent_Charbel(y, tx, initial_w, max_iters, gamma, batch_size)

# Least Squares Regression using Normal Equations
def least_squares(y, tx):
    return least_squares_Charbel(y, tx)

# Ridge Regression using Normal Equations
def ridge_regression(y, tx, lambda_):
    return ridge_regression_Charbel(y, tx, lambda_)

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
    y[y < 0.5] = 0
    y[y >= 0.5] = 1

    mse_test = compute_MSE(y, tx, w)
    print(y)
    print(w)
    print(mse_test)

    create_submission(ids, y, "out.txt")

runModel()



