import csv
import numpy as np
from helpers_higgs import *
import datetime
from implementations import *

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


if __name__ == '__main__':
    # load project data
    features, output, ids = load_training_data()
    y = output
    tx = build_model_data(features)

    # set up testing parameters
    max_iters = 100
    gamma = 0.05
    batch_size = 1
    lambda_ = 0.5
    w_initial = np.array([0] * 31)

    # train
    start_time = datetime.datetime.now()
    w, loss = least_squares(y, tx)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("SGD: execution time={t:.7f} seconds".format(t=exection_time))
    print("optimal weights: ", w)
    print("mse: ", loss)

    # load test data
    test_features, _, test_ids = load_test_data()
    tx = build_model_data(test_features)
    predictions = tx.dot(w)
    predictions[predictions < 0] = -1
    predictions[predictions >= 0] = 1

    # make submission csv
    create_csv_submission(test_ids, predictions, "least_squares_1.csv")
