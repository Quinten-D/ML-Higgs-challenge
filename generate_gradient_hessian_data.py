from implementations import *
from helpers_higgs import *
from hessian_logistic_regression import *
#from helpers import *


### handy functions ###
def train_model(y, tx, y_test, tx_test, initial_weights, max_iters, gamma):
    test_loss_list = []
    ### train the model ###
    w = initial_weights
    N = len(y)
    print("START TRAINING")
    for n_iter in range(max_iters):
        #print("iteration:  ", n_iter)
        if n_iter % 10 == 0:
            loss = compute_log_loss(y_test, tx_test, w)
            test_loss_list.append(loss)
        gradient = compute_gradient_log_loss(y, tx, w)
        w = w - (gamma * gradient)
    # compute log loss
    loss = compute_log_loss(y, tx, w)
    print("final training loss: ", loss)
    final_test_loss = compute_log_loss(y_test, tx_test, w)
    print("final test loss: ", final_test_loss)
    test_loss_list.append(final_test_loss)
    acc = accuracy(y_test, tx_test, w)
    print("accuracy on test data: ", acc)
    print("test loss list GRADIENT: ", test_loss_list, "\n")
    return test_loss_list


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
    max_iters = 500
    gamma = 0.001

    ### train the model, take the average of its test loss ###
    average_test_loss_list = np.zeros(int(max_iters/10)+1)
    nb_of_samples = 1
    for i in range(1):
        print("Iteration: ", i+1)
        average_test_loss_list += np.array(train_model(y, tx, y_test, tx_test, initial_weights, max_iters, gamma))
    average_test_loss_list /= nb_of_samples
    print("AVERAGE TEST LOSS LIST GRADIENT: ", average_test_loss_list)

