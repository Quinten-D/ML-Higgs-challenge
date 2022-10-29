from implementations import *
from helpers_higgs import *
#from helpers import *


### handy functions ###
def load_csv_data_logistic(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]
    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = 0
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

def preproces(input_data):
    features = []
    for i in range(0, 30):
        curFeature = input_data[:, i]
        curFeature[curFeature == -999] = np.mean(curFeature[curFeature != -999])
        standardize(curFeature)
        features.append(curFeature)
    return np.array(features).T

def accuracy(y, tx, w):
    predictions = sigmoid(np.dot(tx, w))
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    difference = y-predictions
    mistakes = np.count_nonzero(difference)
    return (len(y)-mistakes)/len(y)

def compute_Hessian(tx, w):
    N = len(tx)
    # compute diagonal matrix S
    diagonal = sigmoid(tx.dot(w)) * (np.ones(N)-sigmoid(tx.dot(w)))
    s = np.diag(diagonal)
    return np.dot(tx.T, np.dot(s, tx))  # theoretically this needs to be multiplied by 1/N


if __name__ == '__main__':
    ### load project data ###
    output, features, ids = load_csv_data_logistic("train.csv", sub_sample=False)
    y = output
    #features = preproces(features)
    #tx = build_model_data(features)
    #print(tx.shape)
    tx = build_model_data(standardize(features)[0])
    # split labeled data into 80% training data and 20% test data
    y_test = y[200000:]
    y = y[:200000]
    tx_test = tx[200000:]
    tx = tx[:200000]

    ### define the hyperparameters for logistic regression ###
    nb_of_parameters = len(tx[0]) # should be 31 normally, 30 features + 1 bias term
    initial_weights = np.array([0.46756248, 0.82084076, 0.13473604, 0.06748474, 0.08071737,
                                0.89997862, 0.99040634, 0.88295851, 0.56703793, 0.25140082,
                                0.81367198, 0.48045343, 0.26640933, 0.90796936, 0.48122395,
                                0.77356115, 0.55607271, 0.96981431, 0.29737622, 0.90175285,
                                0.02513868, 0.08031006, 0.5847512 , 0.13558202, 0.35724844,
                                0.79922558, 0.40078367, 0.20064134, 0.22376159, 0.64714853,
                                0.63752236])
    max_iters = 10
    gamma = 0.001
    batch_size = 128

    ### train the model ###
    w = initial_weights
    N = len(y)
    print("START TRAINING")
    for n_iter in range(max_iters):
        print("iteration:  ", n_iter)
        loss = compute_log_loss(y_test, tx_test, w)
        print("test loss: ", loss, "\n")
        # choose batch_size data points
        data_points = np.random.randint(0, N, size=batch_size)
        # pick out the datapoints
        x_batch = tx[data_points]
        y_batch = y[data_points]
        # compute gradient
        gradient = compute_gradient_log_loss(y, tx, w)
        # compute stochastic Hessian
        stochastic_hessian = compute_Hessian(x_batch, w)
        # update w by matrix product of inverse Hessian and gradient
        # because hessian wasn't multiplied with 1/batch size (for practical reasons), the inverse Hessian is
        # actually 1/batch_size * inverse Hessian, therefore we need to multiply it with batch_size to get the actual inverse Hessian
        w = w - (gamma * batch_size * np.dot(np.linalg.inv(stochastic_hessian), gradient))
    # compute log loss
    loss = compute_log_loss(y, tx, w)
    print("final training loss: ", loss)

    ### test the model ###
    test_loss = compute_log_loss(y_test, tx_test, w)
    accuracy = accuracy(y_test, tx_test, w)
    print("final test loss: ", test_loss)
    print("accuracy on test data: ", accuracy)
    print("final weights: ", w, "\n")

    ### load the official test data ###
    test_features, _, test_ids = load_test_data()
    tx = build_model_data(standardize(test_features)[0])
    predictions = sigmoid(tx.dot(w))
    predictions[predictions < 0.5] = -1
    predictions[predictions >= 0.5] = 1

    ### make submission csv ###
    #create_csv_submission(test_ids, predictions, "logistic_regression_2.csv")
    #print("csv file made")

