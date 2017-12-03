# File name: MoonsModel
# Copyright 2017 Chidambaram Periakaruppan
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets

from Model import Model
from NN import NN

# from NNTF import NNTF
# from NNKeras import NNKeras as NN

np.random.seed(1)


class MoonsModel(Model):
    '''
    Class implementation for moons model.
    '''

    def load_data(self, train_path, test_path):
        '''

        Loads the Moons data.

        Args:
            data_path(str): Ddata file path.


        Returns:

        '''
        np.random.seed(3)
        train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)

        # test_X = data['Xval']
        # test_Y = data['yval']
        return train_X, train_Y, None, None, None

    def transform_data(self):
        '''
        Does nothing for the moons dataset.
        Returns:
            tuple: train_set_x_orig(ndarray): Transformed training data.
            test_set_x_orig(ndarray): Transformed test data.

        '''
        return self.train_X_orig, self.test_X_orig

    def show_errors(self):
        '''
        Shows the errors.

        '''
        super(MoonsModel, self).show_errors()

        classification = self.test_Y[self.errors]
        prediction = self.test_Y_pred[self.errors]
        images = self.test_X_orig[self.errors]

        self.show_data(images, classification)

    def show_data(self, X=np.array([]), Y=np.array([])):
        '''

        Shows the data.

        Args:
            X: X of dataset.
            Y: Y of dataset.


        '''

        if X.shape[0] == 0:
            X = self.train_X_orig
        if Y.shape[0] == 0:
            Y = self.train_Y
        classes = self.classes

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

    Runs the coursera unit test for the moons dataset.


    '''

    moons_model = MoonsModel('', '', True)
    # moons_model.show_data()
    L = [2, 5, 2, 1]
    activations = ['relu', 'relu', 'sigmoid']
    lambd = 0.
    learning_rate = 0.0007
    mini_batch_size = 64
    beta1 = 0
    epochs = 10000
    keep_prob = 1.

    gradient_check = False
    print_cost = False
    init_method = 'he'

    moons_model.train(NN, L, activations, lambd, keep_prob, learning_rate, init_method, epochs, mini_batch_size, 3,
                      gradient_check,
                      print_cost=print_cost, beta1=beta1)
    # moons_model.plot_decision_boundary()
    moons_model.predict_train()

    expected_result = {'J': 0.52344892570930301, 'train': 0.7966666666666666}
    print('Moons model result 1', moons_model.unit_test)
    print('Moons model expected 1', expected_result)
    if moons_model.unit_test == expected_result:
        print("Moons model unit test 1: OK!!!")
    else:
        print("Moons model results don't match expected results. Please check!!!")

    moons_model = MoonsModel('', '', True)
    # moons_model.show_data()

    beta1 = 0.9

    moons_model.train(NN, L, activations, lambd, keep_prob, learning_rate, init_method, epochs, mini_batch_size, 3,
                      gradient_check,
                      print_cost=print_cost, beta1=beta1)
    # moons_model.plot_decision_boundary()
    moons_model.predict_train()

    expected_result = {'J': 0.52461069011408223, 'train': 0.7966666666666666}
    print('Moons model result 2', moons_model.unit_test)
    print('Moons model expected 2', expected_result)
    if moons_model.unit_test == expected_result:
        print("Moons model unit test 2: OK!!!")
    else:
        print("Moons model results don't match expected results. Please check!!!")

    moons_model = MoonsModel('', '', True)
    # moons_model.show_data()


    beta1 = 0.9
    beta2 = 0.999
    correction = True

    moons_model.train(NN, L, activations, lambd, keep_prob, learning_rate, init_method, epochs, mini_batch_size, 3,
                      gradient_check,
                      print_cost=print_cost, beta1=beta1, beta2=beta2, correction=correction)

    # moons_model.plot_decision_boundary()
    moons_model.predict_train()

    expected_result = {'J': 0.048886696890171777, 'train': 0.94}
    print('Moons model result 3', moons_model.unit_test)
    print('Moons model expected 3', expected_result)
    if moons_model.unit_test == expected_result:
        print("Moons model unit test 3: OK!!!")
    else:
        print("Moons model results don't match expected results. Please check!!!")
    return

    return


if __name__ == "__main__":
    unit_test()
