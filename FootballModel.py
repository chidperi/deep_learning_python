# File name: FootballModel
# Copyright 2017 Chidambaram Periakaruppan
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from Model import Model
from NN import NN

# from NNTF import NNTF
# from NNKeras import NNKeras as NN

np.random.seed(1)


class FootballModel(Model):
    '''
    Class implementation for football model.
    '''

    def load_data(self, train_path, test_path):
        '''

        Loads the football data given the path.

        Args:
            data_path(str): Ddata file path.


        Returns:

        '''

        data = scipy.io.loadmat(train_path)
        train_X = data['X']
        train_Y = data['y']
        test_X = data['Xval']
        test_Y = data['yval']
        return train_X, train_Y, test_X, test_Y, None

    def transform_data(self):
        '''
        Does nothing for the football dataset.
        Returns:
            tuple: train_set_x_orig(ndarray): Transformed training data.
            test_set_x_orig(ndarray): Transformed test data.

        '''
        return self.train_X_orig, self.test_X_orig

    def show_errors(self):
        '''
        Shows the errors.

        '''
        super(FootballModel, self).show_errors()

        classification = self.test_Y[self.errors]
        prediction = self.test_Y_pred[self.errors]
        images = self.test_X_orig[self.errors]

        self.show_data(images, classification)

    def show_data(self, X=np.array([]), Y=np.array([])):
        '''

        Shows the positions that the ball was won and lost.

        Args:
            X: X of dataset.
            Y: Y of dataset.


        '''

        if X.shape[0] == 0:
            X = self.train_X_orig
        if Y.shape[0] == 0:
            Y = self.train_Y
        classes = self.classes

        plt.rcParams['figure.figsize'] = (7.0, 4.0)
        plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), s=40, cmap=plt.cm.Spectral)
        plt.show()

    def plot_decision_boundary(self):
        '''
        Plots the points and decision boundary after the training.

        '''

        m, features = self.train_X_orig.shape
        if features != 2:
            raise ValueError('Only 2 feature decision boundaries can be plotted')
        ranges = []
        for i in range(0, features):
            min_X = np.min(self.train_X_orig[:, i])
            max_X = np.max(self.train_X_orig[:, i])
            range_X = np.linspace(min_X, max_X)
            ranges.append(range_X)

        xx, yy = np.meshgrid(ranges[0], ranges[1])

        X_mesh = np.c_[xx.flatten(), yy.flatten()]

        Y_pred, _ = self.neural_net.predict(X_mesh)

        plt.contourf(xx, yy, Y_pred.reshape(50, 50), cmap=plt.cm.Spectral)
        self.show_data()


def unit_test():
    '''

    Runs the coursera unit test for the football dataset.

    '''

    football_model = FootballModel('./dataset/football/data.mat', '', True)
    # football_model.show_data()
    L = [2, 20, 3, 1]
    activations = ['relu', 'relu', 'sigmoid']
    lambd = 0.
    keep_prob = 1.
    learning_rate = 0.3
    mini_batch_size = None
    epochs = 30000
    gradient_check = False
    print_cost = False
    init_method = 'xavier'

    football_model.train(NN, L, activations, lambd, keep_prob, learning_rate, init_method, epochs, mini_batch_size, 3,
                         gradient_check,
                         print_cost=print_cost)
    # football_model.plot_decision_boundary()
    football_model.predict_train()
    football_model.predict_test()
    # football_model.show_errors()
    #

    expected_result = {'J': 0.12509131245510335, 'train': 0.9478672985781991, 'test': 0.915}

    print('Football model result 1', football_model.unit_test)
    print('Football model expected 1', expected_result)
    if football_model.unit_test == expected_result:
        print("Football model unit test 1: OK!!!")
    else:
        print("Football model results don't match expected results. Please check!!!")

    football_model = FootballModel('./dataset/football/data.mat', '', True)
    # football_model.show_data()
    L = [2, 20, 3, 1]
    activations = ['relu', 'relu', 'sigmoid']
    lambd = 0.7
    keep_prob = 1.
    learning_rate = 0.3

    football_model.train(NN, L, activations, lambd, keep_prob, learning_rate, init_method, epochs, mini_batch_size, 3,
                         gradient_check,
                         print_cost=print_cost)
    # football_model.plot_decision_boundary()
    football_model.predict_train()
    football_model.predict_test()
    # football_model.show_errors()
    expected_result = {'J': 0.2678617428709586, 'train': 0.9383886255924171, 'test': 0.93}

    print('Football model result 2', football_model.unit_test)
    print('Football model expected 2', expected_result)
    if football_model.unit_test == expected_result:
        print("Football model unit test 2: OK!!!")
    else:
        print("Football model results don't match expected results. Please check!!!")

    football_model = FootballModel('./dataset/football/data.mat', '', True)
    # football_model.show_data()
    L = [2, 20, 3, 1]
    activations = ['relu', 'relu', 'sigmoid']
    lambd = 0.
    keep_prob = [0.86, 0.86, 1.]

    learning_rate = 0.3

    football_model.train(NN, L, activations, lambd, keep_prob, learning_rate, init_method, epochs, mini_batch_size, 3,
                         gradient_check,
                         print_cost=print_cost)
    # football_model.plot_decision_boundary()
    football_model.predict_train()
    football_model.predict_test()
    # football_model.show_errors()
    expected_result = {'J': 0.06048817515244604, 'train': 0.9289099526066351, 'test': 0.95}

    print('Football model result 3', football_model.unit_test)
    print('Football model expected 3', expected_result)
    if football_model.unit_test == expected_result:
        print("Football model unit test 3: OK!!!")
    else:
        print("Football model results don't match expected results. Please check!!!")


if __name__ == "__main__":
    unit_test()
