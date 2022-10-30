from implementations import *
from helpers import *
from utils import *

def build_k_indices(y, k_fold, seed):
    """Builds k indices for k-fold cross validation.
    
    Args:
        y:      shape=(N,)
        k_fold: number of partitions
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def lambda_grid_search(y_train, x_train, y_test, x_test, lambda_grid, max_iters, gamma):
    """Finds the hyperparameter and weights associated with the lowest test loss within a specified grid

        Args:
            y_train:        shape=(N - N/k_fold,)
            x_train:        shape=(N - N/k_fold, D)
            y_test:         shape=(N/k_fold,)
            x_test:         shape=(N/k_fold, D)
            lambda_grid:    list of values for the regularization parameter
            max_iters:      number of iterations of gradient descent
            gamma:          step size of gradient descent

        Returns:
            weight:         shape=(D,)
            lambda_:        scalar, regularization parameter associated with the lowest test loss
    """
    weight_list = []
    loss_list   = []

    for lambda_ in lambda_grid: 
        weight = reg_logistic_regression(y_train, x_train, lambda_, max_iters, gamma)
        loss   = compute_log_loss(y_test, x_test, weight, lambda_)
        weight_list.append(weight)
        loss_list.append(loss)

    optimal_weight, optimal_lambda = weight_list[np.argmin(loss_list)], lambda_grid[np.argmin(loss_list)]
    return optimal_weight, optimal_lambda


def cross_validation_tuning(y, x, k_fold, lambda_grid, max_iters, gamma):
    """Uses lambda_grid_search and cross validation to tune hyperparameter lambda

        Args:
            y:              shape=(N,)
            x:              shape=(N,D)
            k_fold:         number of partitions in the cross validation
            lambda_grid:    list of values for the regularization parameter
            max_iters:      number of iterations of gradient descent
            gamma:          step size of gradient descent

        Returns:
            weight:         shape=(D,)
            lambda_:        scalar, regularization parameter associated with the lowest test loss
    """
    # Get fold indices for cross validation
    indices = build_k_indices(y, k_fold, 1)

    weight_lambda_pairs = []
    for k in range(k_fold):
        # Holdout kth fold for testing and train on the rest
        x_test,  y_test  = x[indices[k]], y[indices[k]]
        rest_indices = np.delete(indices, k, axis=0)
        x_train, y_train = x[rest_indices.ravel()], y[rest_indices.ravel()]

        # Perfom grid search for optimal lambda
        optimal_weight, optimal_lambda = lambda_grid_search(y_train, x_train, y_test, x_test, lambda_grid, max_iters, gamma)

        # Store optimal weight and lambda
        weight_lambda_pairs.append((optimal_weight, optimal_lambda))

        indices = build_k_indices(y, k_fold, 1)

    # iterate over the fold results and compute loss
    loss_list = []
    for weight, lambda_ in weight_lambda_pairs:
        loss = compute_log_loss(y, x, weight, lambda_)
        loss_list.append(loss)
    
    # return (weight, lambda) pair with the lowest loss
    return weight_lambda_pairs[np.argmin(loss_list)]