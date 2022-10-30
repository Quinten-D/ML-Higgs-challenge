from utils import *
from implementations import *
from hessian_logistic_regression import accuracy, compute_Hessian


def train_model_least_squares(yb, tx):
    """
    Trains the model using Least Squares
    Returns the trained weights, the MSE
    """

    return least_squares(yb, tx)


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
        ]
    )

    initial_weights = initial_weights[: tx.shape[1]]

    max_iters = 5000
    gamma = 0.01

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
        ]
    )

    initial_weights = initial_weights[: tx.shape[1]]

    max_iters = 2000
    # gamma = 0.01
    # batch_size = 612

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
        w = w - (
            gamma * batch_size * np.dot(np.linalg.inv(stochastic_hessian), gradient)
        )
    # compute log loss
    loss = compute_log_loss(yb, tx, w)

    return w, loss


def train_model():
    """
    Trains the model on 3 subsets of the data
    Returns for each subset:
    the trained weights, the features that were not deemed useful, the mean of the kept features
    """

    (
        (yb0, processed_data0, removed_features0, means0, stds0),
        (yb1, processed_data1, removed_features1, means1, stds1),
        (yb23, processed_data23, removed_features23, means23, stds23),
    ) = load_training_data(using_logistic_regression=True)

    all_w = []
    all_removed_features = []
    all_means = []
    for (yb, processed_data, removed_features, means) in [
        (yb0, processed_data0, removed_features0, means0),
        (yb1, processed_data1, removed_features1, means1),
        (yb23, processed_data23, removed_features23, means23),
    ]:
        tx = build_model_data(processed_data)
        # split into train and test data
        index = 4 * len(yb) // 5
        yb_test = yb[index:]
        yb = yb[:index]
        tx_test = tx[index:]
        tx = tx[:index]
        # train
        print("Start Training")
        print(yb.shape, tx.shape, yb_test.shape, tx_test.shape)
        w, loss = train_model_hessian_logistic_regression(
            yb, tx, gamma=0.0005, batch_size=612
        )
        # w, loss = train_model_logistic_regression(yb, tx)

        # test trained model on test data
        test_loss = compute_log_loss(yb_test, tx_test, w)
        acc = accuracy(yb_test, tx_test, w)
        print("final train loss: ", loss)
        print("final test loss: ", test_loss)
        print("accuracy on test data: ", acc)
        print("final weights: ", w, "\n")

        all_w.append(w)
        all_removed_features.append(removed_features)
        all_means.append(means)

    return all_w, all_removed_features, all_means


def runModel():
    """
    Trains the model and then run it on a test set to predict the results
    """

    all_w, all_removed_features, all_means = train_model()
    all_processed_data, all_ids = load_test_data(all_removed_features, all_means)

    id_prediction_pairs = []
    for i in range(3):
        processed_data = all_processed_data[i]
        ids = all_ids[i]
        w = all_w[i]

        tx = build_model_data(processed_data)

        predictions = sigmoid(tx.dot(w))
        predictions[predictions < 0.5] = -1
        predictions[predictions >= 0.5] = 1

        for j in range(len(ids)):
            id_prediction_pairs.append((ids[j], predictions[j]))

        print("Done with " + str(i))

    id_prediction_pairs.sort()
    ids = []
    predictions = []
    for j in range(len(id_prediction_pairs)):
        ids.append(id_prediction_pairs[j][0])
        predictions.append(id_prediction_pairs[j][1])

    create_csv_submission(ids, predictions, "out2.txt")


if __name__ == "__main__":
    train_model()
