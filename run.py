# -*- coding: utf-8 -*-
from utils import *
from implementations import *

def train_model_least_squares(yb, tx):
    """
    Trains the model using Least Squares
    Returns the trained weights, the MSE
    """

    return least_squares(yb, tx)

def train_model_logistic_regression(yb, tx):
    """
    Trains the model using Logistic Regression
    Returns the trained weights, MSE
    """

    initial_weights = np.array([0.46756248, 0.82084076, 0.13473604, 0.06748474, 0.08071737,
                                0.89997862, 0.99040634, 0.88295851, 0.56703793, 0.25140082,
                                0.81367198, 0.48045343, 0.26640933, 0.90796936, 0.48122395,
                                0.77356115, 0.55607271, 0.96981431, 0.29737622, 0.90175285,
                                0.02513868, 0.08031006, 0.5847512, 0.13558202, 0.35724844,
                                0.79922558, 0.40078367, 0.20064134, 0.22376159, 0.64714853,
                                0.63752236])

    initial_weights = initial_weights[:tx.shape[1]]

    max_iters = 100
    gamma = 0.1

    return logistic_regression(yb, tx, initial_weights, max_iters, gamma)

def train_model():
    """
    Trains the model on 3 subsets of the data
    Returns for each subset:
    the trained weights, the features that were not deemed useful, the mean of the kept features
    """

    (yb0, processed_data0, removed_features0, means0), \
    (yb1, processed_data1, removed_features1, means1), \
    (yb23, processed_data23, removed_features23, means23) = load_training_data(using_logistic_regression=True)

    all_w = []
    all_removed_features = []
    all_means = []
    for (yb, processed_data, removed_features, means) in [(yb0, processed_data0, removed_features0, means0),
                                                          (yb1, processed_data1, removed_features1, means1),
                                                          (yb23, processed_data23, removed_features23, means23)]:
        tx = build_model_data(processed_data)
        w, loss = train_model_logistic_regression(yb, tx)

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


runModel()
