
import numpy as np
from helpers import batch_iter

def compute_MSE(y, tx, w):
    N = y.shape[0]
    e = y - np.dot(tx, w)
    return 1 / (2 * N) * np.sum(e ** 2)

def gradient_descent_Charbel(y, tx, initial_w, max_iters, gamma):
    def compute_gradient(y, tx, w):
        N = y.shape[0]
        e = y - np.dot(tx, w)
        return -1 / N * np.dot(tx.transpose(), e)

    ws = []
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = compute_MSE(y, tx, w)

        ws.append(w)
        losses.append(loss)

        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def stochastic_gradient_descent_Charbel(y, tx, initial_w, batch_size, max_iters, gamma):
    def compute_stoch_gradient(y, tx, w):
        N = y.shape[0]
        e = y - np.dot(tx, w)
        return -1 / N * np.dot(tx.transpose(), e)

    ws = []
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # Here we assume num_batches is 1; otherwise we need to update n_iter as well
        # This can definitely be optimized since we shuffle everything each time we want one batch
        for curY, curTX in batch_iter(y, tx, batch_size):
            gradient = compute_stoch_gradient(curY, curTX, w)
            w = w - gamma * gradient
            loss = compute_MSE(y, tx, w)

            ws.append(w)
            losses.append(loss)

            print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

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
    return np.dot(np.dot(np.linalg.inv(XtX_Lambda), tx.T), y)


# Linear Regression using Gradient Descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    return gradient_descent_Charbel(y, tx, initial_w, max_iters, gamma)

# Linear Regression using Stochastic Gradient Descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    return stochastic_gradient_descent_Charbel(y, tx, initial_w, max_iters, gamma)

# Least Squares Regression using Normal Equations
def least_squares(y, tx):
    return least_squares_Charbel(y, tx)

# Ridge Regression using Normal Equations
def ridge_regression(y, tx, lambda_):
    return ridge_regression_Charbel(y, tx, lambda_)



