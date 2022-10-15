from implementations import *
from helpers import *
from plots import *


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    N = len(x)
    result_matrix = np.zeros((degree + 1, N))
    for i in range(0, degree + 1):
        result_matrix[i] = np.power(x, i)
    return np.transpose(result_matrix)

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    N = len(x)
    index_order = np.random.permutation(N)
    split_index = int(np.floor(N * ratio))
    training_indexes = index_order[:split_index]
    control_indexes = index_order[split_index:]
    return (x[training_indexes], x[control_indexes], y[training_indexes], y[control_indexes])
    #raise NotImplementedError


def train_test_split_demo_1(x, y, degree, ratio, seed):
    # split the data, and return train and test data:
    train_x, test_x, train_y, test_y = split_data(x, y, ratio, seed)
    # form train and test data with polynomial basis function:
    train_x_poly = build_poly(train_x, degree)
    test_x_poly = build_poly(test_x, degree)
    # calculate weight and loss
    w, loss = mean_squared_error_gd(train_y,train_x_poly, np.random.rand(len(train_x_poly[0])), 100, 0.01)
    print("proportion={p}, degree={d}, Training loss={tr:.3f}".format(
        p=ratio, d=degree, tr=loss))
    return train_x, test_x, train_y, test_y, w

def train_test_split_demo_3(x, y, degree, ratio, seed):
    # split the data, and return train and test data:
    train_x, test_x, train_y, test_y = split_data(x, y, ratio, seed)
    # form train and test data with polynomial basis function:
    train_x_poly = build_poly(train_x, degree)
    test_x_poly = build_poly(test_x, degree)
    # calculate weight and loss
    w, loss = least_squares(train_y, train_x_poly)
    print("proportion={p}, degree={d}, Training loss={tr:.3f}".format(
        p=ratio, d=degree, tr=loss))
    return train_x, test_x, train_y, test_y, w


if __name__ == '__main__':
    # load dataset
    x, y = load_data()
    print("shape of x {}".format(x.shape))
    print("shape of y {}".format(y.shape))

    # demo time, make plot of test data and predictions on test data
    seed = 6
    degrees = [1, 3, 7, 12]
    split_ratios = [0.9, 0.7, 0.5, 0.1]
    # define the structure of the figure
    num_row = 4
    num_col = 4
    axs = plt.subplots(num_row, num_col, figsize=(20, 8))[1]

    for ind, split_ratio in enumerate(split_ratios):
        for ind_d, degree in enumerate(degrees):
            x_tr, x_te, y_tr, y_te, w = train_test_split_demo_3(x, y, degree, split_ratio, seed)
            plot_fitted_curve(
                y_tr, x_tr, w, degree, axs[ind_d][ind % num_col])
            axs[ind_d][ind].set_title(f'Degree: {degree}, Split {split_ratio}')
    plt.tight_layout()
    plt.show()



