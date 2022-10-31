import random
from utils import *
from implementations import *
from hessian_logistic_regression import accuracy, compute_Hessian


def train_model_least_squares(yb, tx):
    """
    Trains the model using Least Squares
    Returns the trained weights, the MSE
    """

    return least_squares(yb, tx)


def train_model_ridge_regression(yb, tx, lambda_):
    return ridge_regression(yb, tx, lambda_)


def accuracy_mse(y, tx, w):
    predictions = np.dot(tx, w)
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    difference = y - predictions
    mistakes = np.count_nonzero(difference)
    return (len(y) - mistakes) / len(y)


def accuracy_and_mistakes(y, tx, w):
    predictions = sigmoid(np.dot(tx, w))
    #predictions = np.dot(tx, w)
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    difference = y - predictions
    mistakes = np.count_nonzero(difference)
    return (len(y) - mistakes) / len(y), mistakes


def accuracy_and_mistakes_random(y):
    mistakes = 0
    for i in y:
        guess = random.randint(0,2)
        #guess = abs(1-guess)
        if guess == i:
            mistakes+=1
    return (len(y) - mistakes) / len(y), mistakes


def train_model_logistic_regression(yb, tx):
    """
    Trains the model using Newton's method with a stochastic Hessian matrix
    Returns the trained weights, log loss
    """

    initial_weights = np.array(
        [
            0.46756248,
            0.82084076,
            0.13473604,
            0.06748474,
            0.08071737,
            0.89997862,
            0.99040634,
            0.88295851,
            0.56703793,
            0.25140082,
            0.81367198,
            0.48045343,
            0.26640933,
            0.90796936,
            0.48122395,
            0.77356115,
            0.55607271,
            0.96981431,
            0.29737622,
            0.90175285,
            0.02513868,
            0.08031006,
            0.5847512,
            0.13558202,
            0.35724844,
            0.79922558,
            0.40078367,
            0.20064134,
            0.22376159,
            0.64714853,
            0.63752236,
            0.46756248,
            0.82084076,
            0.13473604,
            0.06748474,
            0.08071737,
            0.89997862,
            0.99040634,
            0.88295851,
            0.56703793,
            0.25140082,
            0.81367198,
            0.48045343,
            0.26640933,
            0.90796936,
            0.48122395,
            0.77356115,
            0.55607271,
            0.96981431,
            0.29737622,
            0.90175285,
            0.02513868,
            0.08031006,
            0.5847512,
            0.13558202,
            0.35724844,
            0.79922558,
            0.40078367,
            0.20064134,
            0.22376159,
            0.64714853,
            0.63752236,
        ]
    )
    initial_weights = initial_weights[: tx.shape[1]]

    max_iters = 300
    gamma = 0.1

    return logistic_regression(yb, tx, initial_weights, max_iters, gamma)


def train_model_hessian_logistic_regression(yb, tx, gamma, batch_size):
    """
    Trains the model using Newton's method with a stochastic Hessian matrix
    Returns the trained weights, log loss
    """

    initial_weights = np.array(
        [
            0.46756248,
            0.82084076,
            0.13473604,
            0.06748474,
            0.08071737,
            0.89997862,
            0.99040634,
            0.88295851,
            0.56703793,
            0.25140082,
            0.81367198,
            0.48045343,
            0.26640933,
            0.90796936,
            0.48122395,
            0.77356115,
            0.55607271,
            0.96981431,
            0.29737622,
            0.90175285,
            0.02513868,
            0.08031006,
            0.5847512,
            0.13558202,
            0.35724844,
            0.79922558,
            0.40078367,
            0.20064134,
            0.22376159,
            0.64714853,
            0.63752236,
            0.46756248,
            0.82084076,
            0.13473604,
            0.06748474,
            0.08071737,
            0.89997862,
            0.99040634,
            0.88295851,
            0.56703793,
            0.25140082,
            0.81367198,
            0.48045343,
            0.26640933,
            0.90796936,
            0.48122395,
            0.77356115,
            0.55607271,
            0.96981431,
            0.29737622,
            0.90175285,
            0.02513868,
            0.08031006,
            0.5847512,
            0.13558202,
            0.35724844,
            0.79922558,
            0.40078367,
            0.20064134,
            0.22376159,
            0.64714853,
            0.63752236,
        ]
    )

    initial_weights = initial_weights[: tx.shape[1]]

    max_iters = 200

    # train the model
    w = initial_weights
    N = len(yb)
    for n_iter in range(max_iters):
        # choose batch_size data points
        data_points = np.random.randint(0, N, size=batch_size)
        # pick out the datapoints
        x_batch = tx[data_points]
        y_batch = yb[data_points]
        # compute gradient
        gradient = compute_gradient_log_loss(yb, tx, w)
        # compute stochastic Hessian
        stochastic_hessian = compute_Hessian(x_batch, w)
        # update w by matrix product of inverse Hessian and gradient
        # because hessian wasn't multiplied with 1/batch size (for practical reasons), the inverse Hessian is
        # actually 1/batch_size * inverse Hessian, therefore we need to multiply it with batch_size to get the actual inverse Hessian
        w = w - (gamma * batch_size * np.dot(np.linalg.pinv(stochastic_hessian), gradient))
    # compute log loss
    loss = compute_log_loss(yb, tx, w)

    return w, loss


def get_accuracy():
    """
    Trains the model on 3 subsets of the data, prints the accuracy on the test data
    Returns for each subset:
    the trained weights, the features that were not deemed useful, the mean of the kept features
    """

    (
        (yb0, processed_data0, removed_features0, means0, stds0),
        (yb1, processed_data1, removed_features1, means1, stds1),
        (yb23, processed_data23, removed_features23, means23, stds23),
    ) = load_training_data(using_logistic_regression=True)

    # 20% test data
    nb_of_test_points = 1 * (len(yb0)+len(yb1)+len(yb23)) // 5
    nb_of_test_point_0 = random.randint(0, nb_of_test_points)
    nb_of_test_point_1 = random.randint(0, nb_of_test_points-nb_of_test_point_0)
    nb_of_test_point_23 = nb_of_test_points - nb_of_test_point_0 - nb_of_test_point_1
    nb_of_points = [nb_of_test_point_0, nb_of_test_point_1, nb_of_test_point_23]
    print(nb_of_test_points)
    print(nb_of_points)


    all_w = []
    all_removed_features = []
    all_means = []
    all_stds = []
    counter = 0
    total_mistakes = 0
    for (yb, processed_data, removed_features, means, stds) in [
        (yb0, processed_data0, removed_features0, means0, stds0),
        (yb1, processed_data1, removed_features1, means1, stds1),
        (yb23, processed_data23, removed_features23, means23, stds23),
    ]:
        tx = build_model_data(processed_data)
        # add features
        D = len(tx[0])
        N = len(tx)
        for feature_col in range(1, D):
            tx = np.append(tx, (tx[:, feature_col].reshape((N, 1))) ** 2, axis=1)
        # split into train and test data
        #index = 4 * len(yb) // 5
        index = nb_of_points[counter]
        test_indices = random.sample(range(0, len(tx)), index)
        counter += 1
        yb_test = yb[test_indices]
        yb = np.delete(yb, test_indices)
        tx_test = tx[test_indices]
        tx = np.delete(tx, test_indices, axis=0)
        # train
        print("Start Training")
        print(yb.shape, tx.shape, yb_test.shape, tx_test.shape)
        w, loss = train_model_hessian_logistic_regression(yb, tx, gamma=0.01, batch_size=128)
        #w, loss = train_model_logistic_regression(yb, tx)
        #w, loss = least_squares(yb, tx)
        #w, loss = train_model_ridge_regression(yb, tx, 0.)

        # test trained model on test data
        test_loss = compute_log_loss(yb_test, tx_test, w)
        acc, m = accuracy_and_mistakes(yb_test, tx_test, w)
        #acc, m = accuracy_and_mistakes_random(yb_test)
        print("m ", m)
        total_mistakes += m
        print("final train loss: ", loss)
        print("final test loss: ", test_loss)
        print("accuracy on test data: ", acc)
        print("final weights: ", w, "\n")

        all_w.append(w)
        all_removed_features.append(removed_features)
        all_means.append(means)
        all_stds.append(stds)

    # global accuracy
    global_accuracy = (nb_of_test_points-total_mistakes)/nb_of_test_points
    print("global accuracy: ", global_accuracy)


    return all_w, all_removed_features, all_means, all_stds



if __name__ == "__main__":
    get_accuracy()
    #runModel()
