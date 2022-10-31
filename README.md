# Machine Learning - Project 1
This repository contains the codes used to solve the ML Higgs challenge.

The files in the root directory contain the final codes used for the submission. The "Testing Files" directory contains scripts used to test multiple methods.

## helpers.py
Contains function to manipulate the data:
- `load_csv_data`: Reads the data file
- `create_csv_submission`: Writes the output on file
- `standardize`: Standardizes the features
- `build_model_data`: Adds a column of 1s to the features
- `batch_iter`: Generate a mini-batch iterator for a dataset

## implementations.py
Contains the 6 required implementations for this project:
- `mean squared error gd`: Linear regression using gradient descent
- `mean squared error sgd`: Linear regression using stochastic gradient descent
- `least_squares`: Least squares regression using normal equations
- `ridge_regression`: Ridge regression using normal equations
- `logistic_regression`: using stochastic gradient descent
- `reg_logistic_regression`: Regularized logistic regression

## run.py
Contains the main function to train and predict:
- `train_model_least_squares`, `train_model_logistic_regression`, `train_Hessian`: Methods to call specific ML training algorithms
- `train_model`: Trains a model and return the parameters
- `runModel`: Runs the model on the testing data

## tuned_logistic.py
Contains code for cross-validation and hyper-parameter optimization:
