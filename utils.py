from helpers import *
import matplotlib as plt


def preprocess_train_data(input_data):
    """
    Does 3 things:
    i) Finds and removes unwanted features (if they have too many missing points or std = 0)
    ii) Replaces -999 with the mean of non -999 values
    ii) Standardizes the data

    Returns the standardized data, the removed features, the means and stds
    """
    processed_data = []
    removed_features = {}
    means = []
    stds = []

    num_samples = input_data.shape[0]
    for i in range(input_data.shape[1]):
        cur_feature = input_data[:, i]

        # Remove features with a lot of missing entries
        unavailable_cnt = np.sum(cur_feature == -999)
        if unavailable_cnt * 1.5 > num_samples or np.std(cur_feature) == 0:
            removed_features[i] = 1
            continue

        # Replace missing entries with the mean of available entries then standardize
        cur_feature[cur_feature == -999] = np.mean(cur_feature[cur_feature != -999])

        means.append(np.mean(cur_feature))
        stds.append(np.std(cur_feature))

        standardize(cur_feature)

        processed_data.append(cur_feature)

    return np.array(processed_data).T, removed_features, means, stds


def preprocess_test_data(input_data, removed_features, means, stds):
    """
    Does 3 things:
    i) Removes unwanted features
    ii) Replaces -999 with the mean found in the training phase
    ii) Standardizes the data

    Returns:
        Processed data
    """
    processed_data = []

    realI = 0
    for i in range(input_data.shape[1]):
        if i in removed_features:
            continue

        cur_feature = input_data[:, i]

        # Replace missing entries with the mean of available entries then standardize
        cur_feature[cur_feature == -999] = means[realI]
        cur_feature -= means[realI]
        cur_feature /= stds[realI]
        realI += 1

        processed_data.append(cur_feature)

    return np.array(processed_data).T


def add_features(input_data):
    D = len(input_data[0])
    N = len(input_data)
    for feature_col in range(1, D):
        input_data = np.append(input_data, (input_data[:, feature_col].reshape((N, 1))) ** 2, axis=1)
    return input_data

def load_training_data(using_logistic_regression=False):
    """
    Args:
        using_logistic_regression: if true, sets the outputs to be in {0,1}; otherwise, keep them as {-1,1}

    Loads the training data and separates it into 3 subsets
    One for PRI_jet_num=0, one for PRI_jet_num=1, one for PRI_jet_num=2 or 3
    Then, preprocess each subset independently

    Returns:
        outputs, processed_data, a set of removed features and an array of means for each subset
    """

    yb, input_data, ids = load_csv_data("Data/train.csv")

    if using_logistic_regression:
        yb[yb == -1] = 0

    # Divide dataset into 3 groups according to PRI_jet_num
    yb0, yb1, yb23 = [], [], []
    input_data0, input_data1, input_data23 = [], [], []

    num_samples = input_data.shape[0]
    for i in range(num_samples):
        if input_data[i][22] == 0:
            yb0.append(yb[i])
            input_data0.append(input_data[i])
        elif input_data[i][22] == 1:
            yb1.append(yb[i])
            input_data1.append(input_data[i])
        else:
            yb23.append(yb[i])
            input_data23.append(input_data[i])

    yb0 = np.array(yb0)
    yb1 = np.array(yb1)
    yb23 = np.array(yb23)

    input_data0 = np.array(input_data0)
    input_data1 = np.array(input_data1)
    input_data23 = np.array(input_data23)

    processed_data0, removed_features0, means0, stds0 = preprocess_train_data(
        input_data0
    )
    processed_data1, removed_features1, means1, stds1 = preprocess_train_data(
        input_data1
    )
    processed_data23, removed_features23, means23, stds23 = preprocess_train_data(
        input_data23
    )

    return (
        (yb0, processed_data0, removed_features0, means0, stds0),
        (yb1, processed_data1, removed_features1, means1, stds1),
        (yb23, processed_data23, removed_features23, means23, stds23),
    )


def load_test_data(all_removed_features, all_means, all_stds):
    """
    Args:
        all_removed_features: 3 sets of removed features, one for each data subset
        all_means: 3 vectors of means, one for each data subset
        all_stds: 3 vectors of stds, one for each data subset

    Separates the test data into 3 subsets according to PRI_jet_num as for the training
    Process each subset separately

    Returns:
        Processed data and ids for each subset
    """

    _, input_data, ids = load_csv_data("Data/test.csv")

    input_data0, input_data1, input_data23 = [], [], []
    ids0, ids1, ids23 = [], [], []

    num_samples = input_data.shape[0]
    for i in range(num_samples):
        if input_data[i][22] == 0:
            input_data0.append(input_data[i])
            ids0.append(ids[i])
        elif input_data[i][22] == 1:
            input_data1.append(input_data[i])
            ids1.append(ids[i])
        else:
            input_data23.append(input_data[i])
            ids23.append(ids[i])

    input_data0 = np.array(input_data0)
    input_data1 = np.array(input_data1)
    input_data23 = np.array(input_data23)

    all_processed_data = []

    idx = 0
    for cur_input_data in [input_data0, input_data1, input_data23]:
        all_processed_data.append(
            preprocess_test_data(
                cur_input_data, all_removed_features[idx], all_means[idx], all_stds[idx]
            )
        )
        idx += 1

    return all_processed_data, [ids0, ids1, ids23]


def check_missing_values():
    """
    Shows that number of missing values is related to PRI_jet_num
    """
    yb, input_data, ids = load_csv_data("Data/train.csv")

    for i in range(input_data.shape[0]):
        numMissing = np.sum(input_data[i] == -999)
        if numMissing > 0:
            print(numMissing, end=": ")
            print(input_data[i][22])
