# File name: Model
# Copyright 2017 Chidambaram Periakaruppan
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import numpy as np


class Model(object):
    '''
    Class for model objects.

    Attributes:
        train_X(ndarray): X training dataset.
        train_Y(ndarray): Y training dataset.
        neural_net(Object): Neural network class to be initialized later.
        train_accuracy(float): Accuracy for the train dataset.
        errors(ndarray): Errors in test dataset.
        test_Y_pred(ndarray): Y predictions for test dataset.
        test_accuracy(float): Accuracy for the test dataset.
        unit_test(dict): Unit test results dictionary.

    '''

    def __init__(self, train_path, test_path, unit_test=False):
        '''

        Args:
            train_path(str): Training data file path.
            test_path(str): Testing data file path.
            unit_test(boolean): Adds a dictionary for unit test results if True. Otherwise None.
        '''

        self.train_X_orig, self.train_Y, self.test_X_orig, self.test_Y, self.classes = self.load_data(train_path,
                                                                                                      test_path)
        self.train_X, self.test_X = self.transform_data()
        self.neural_net = None
        self.train_accuracy = None
        self.errors = None
        self.test_Y_pred = None
        self.test_accuracy = None
        self.unit_test = {} if unit_test else None

    def load_data(self, train_path, test_path):
        '''
        Loads the data given the paths.

        Args:
            train_path(str): Training data file path.
            test_path(str): Testing data file path.

        Returns:

        '''
        return

    def transform_data(self):
        return

    def show_errors(self):
        '''

        Returns:
            ndarray: all errors where the prediction is different from the label.

        '''
        X = self.test_X_orig
        Y = self.test_Y
        Y_pred = self.test_Y_pred
        classes = self.classes

        wrong = Y != Y_pred
        wrong = np.squeeze(wrong)

        self.errors = wrong

    def train(self, NN, L, activations, lambd, keep_prob, learning_rate, xavier, num_epochs, mini_batch_size, seed,
              gradient_check,
              print_cost=True, **kwargs):
        '''
        Trains the neural network using the training data.

        Args:
            NN(Object): Neural network class object {NN, NNTF}.
            L(list): List of nodes in each of the layers including the input and output lauer.
            activations(list):  List of activations in the different layers {'relu', 'sigmoid', 'leaky-relu'}.
            lambd(float): If not None or 0, you will get L2 regularization with L2 penalty.
            keep_prob(float): If less than 1.0, dropout regularization will be implemented.
            learning_rate(float): Learning rate.
            xavier(boolean): True for Xavier initialization otherwise random initialization.
            num_epochs(int): Number of iterations.
            mini_batch_size(int): Mini-batch size.
            seed(int): Ramdom number generator seed.
            gradient_check(boolean): Switches off dropout to allow checking gradient with a numerical check.
            print_cost(boolean): True to print cost as you train.

        '''
        X = self.train_X
        Y = self.train_Y
        self.neural_net = NN(L, activations)

        J = self.neural_net.fit(X, Y, lambd, keep_prob, learning_rate, xavier=xavier,
                                num_epochs=num_epochs, mini_batch_size=mini_batch_size, seed=seed,
                                gradient_check=gradient_check, print_cost=print_cost, **kwargs)
        if type(self.unit_test) == dict:
            self.unit_test['J'] = J

        return

    def predict_train(self):
        '''
        Calculates the prediction and accuracy from the training data.

        '''
        X = self.train_X
        Y = self.train_Y

        _, self.train_accuracy = self.neural_net.predict(X, Y)

        if type(self.unit_test) == dict:
            self.unit_test['train'] = self.train_accuracy

    def predict_test(self):
        '''
        Calculates the prediction and accuracy from the test data.


        '''
        X = self.test_X
        Y = self.test_Y

        self.test_Y_pred, self.test_accuracy = self.neural_net.predict(X, Y)

        if type(self.unit_test) == dict:
            self.unit_test['test'] = self.test_accuracy

    def gradient_check(self, lambd, print_gradients=False):
        self.unit_test['grad_diff'] = self.neural_net.gradient_check(self.train_X, self.train_Y, lambd,
                                                                     print_gradients=print_gradients)
