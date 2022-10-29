from helpers import *


def preprocess_train_data(input_data):
    processed_data = []
    removed_features = {}
    means = []

    num_samples = input_data.shape[0]
    for i in range(input_data.shape[1]):
        cur_feature = input_data[:, i]

        # Remove features with a lot of missing entries
        unavailable_cnt = np.sum(cur_feature == -999)
        if unavailable_cnt * 1.5 > num_samples:
            removed_features[i] = 1
            continue


        # Replace missing entries with the mean of available entries then standardize
        curMean = np.mean(cur_feature[cur_feature != -999])
        means.append(curMean)
        cur_feature[cur_feature == -999] = curMean
        standardize(cur_feature)

        processed_data.append(cur_feature)

    return np.array(processed_data).T, removed_features, means


def preprocess_test_data(input_data, removed_features, means):
    processed_data = []

    realI = 0
    for i in range(input_data.shape[1]):
        if i in removed_features:
            continue

        cur_feature = input_data[:, i]

        # Replace missing entries with the mean of available entries then standardize
        cur_feature[cur_feature == -999] = means[realI]
        standardize(cur_feature)
        realI += 1

        processed_data.append(cur_feature)

    return np.array(processed_data).T


def load_training_data(using_logistic_regression=False):
    yb, input_data, ids = load_csv_data("Data/train.csv")
    processed_data, removed_features, means = preprocess_train_data(input_data)

    if using_logistic_regression:
        yb[yb == -1] = 0

    return yb, processed_data, ids, removed_features, means


def load_test_data(removed_features, means):
    yb, input_data, ids = load_csv_data("Data/test.csv")
    processed_data = preprocess_test_data(input_data, removed_features, means)
    return yb, processed_data, ids