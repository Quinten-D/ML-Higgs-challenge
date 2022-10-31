from implementations import *
from helpers import *

# from helpers import *

def accuracy(y, tx, w):
    predictions = sigmoid(np.dot(tx, w))
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    difference = y - predictions
    mistakes = np.count_nonzero(difference)
    return (len(y) - mistakes) / len(y)


def compute_Hessian(tx, w):
    N = len(tx)
    # compute diagonal matrix S
    diagonal = sigmoid(tx.dot(w)) * (np.ones(N) - sigmoid(tx.dot(w)))
    s = np.diag(diagonal)
    return np.dot(
        tx.T, np.dot(s, tx)
    )  # theoretically this needs to be multiplied by 1/N
