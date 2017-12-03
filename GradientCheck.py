# File name: GradientCheck
# Copyright 2017 Chidambaram Periakaruppan 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the 
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import matplotlib.pyplot as plt
import numpy as np

from Model import Model
from NN import NN

np.random.seed(1)


class GCModel(Model):
    '''
    Class implementation for cat model.
    '''

    def load_data(self, train_path, test_path):
        '''

        Loads the cata data given the paths.

        Args:
            train_path(str): Training data file path.
            test_path(str): Testing data file path.

        Returns:

        '''
        np.random.seed(1)
        train_set_x_orig = np.random.randn(4, 3)
        train_set_x_orig = train_set_x_orig.T

        train_set_y_orig = np.array([[1], [1], [0]])

        test_set_x_orig = train_set_x_orig
        test_set_y_orig = train_set_y_orig
        classes = [0, 1]

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def transform_data(self):
        '''
        Transforms the original data so that they are normalized and samples are in rows and features are in columns.
        Returns:
            tuple: train_set_x_orig(ndarray): Transformed training data.
            test_set_x_orig(ndarray): Transformed test data.

        '''

        m = self.train_X_orig.shape[0]
        train_set_x_orig = self.train_X_orig.reshape(m, -1)
        # train_set_x_orig = train_set_x_orig[:100,:]
        m_test = self.test_X_orig.shape[0]
        test_set_x_orig = self.test_X_orig.reshape(m_test, -1)
        return train_set_x_orig, test_set_x_orig

    def show_errors(self, num_errors=5):
        '''
        Shows the errors.

        Args:
            num_errors: Number of errors to show.

        Returns:

        '''
        super(CatModel, self).show_errors()

        show_num_errors = min(num_errors, np.sum(self.errors * 1))

        classification = self.test_Y[self.errors]
        prediction = self.test_Y_pred[self.errors]
        images = self.test_X_orig[self.errors]

        for i in range(0, show_num_errors):
            self.show_data(i, 3, images, classification)
            print('Prediction is %s' % self.classes[prediction[i, 0]])

    def show_data(self, index, size=6, X=np.array([]), Y=np.array([])):
        '''

        Shows picture of a given index from a dataset and it's label.

        Args:
            index(int): Data sample to show.
            size(int): Size of the image.
            X: X of dataset.
            Y: Y of dataset.


        '''

        if X.shape[0] == 0:
            X = self.train_X_orig
        if Y.shape[0] == 0:
            Y = self.train_Y
        classes = self.classes

        plt.rcParams['figure.figsize'] = (size, size)
        plt.imshow(X[index, :])
        plt.show()

        classification = classes[Y[index, 0]]
        print('This is a %s' % classification)


def unit_test():
    '''

    Runs the coursera unit test for the cat dataset.

    This should print:
    This is a cat Accuracy is 98.5645933014% Accuracy is 80.0%


    '''

    gc_model = GCModel('', '', unit_test=True)
    # cat_model.show_data(2)
    L = [4, 5, 3, 1]
    activations = ['relu', 'relu', 'sigmoid']
    L2 = 0
    keep_prob = 1.
    learning_rate = 0.0075
    iterations = 1
    gradient_check = True
    print_cost = False
    xavier = False

    gc_model.train(NN, L, activations, L2, keep_prob, learning_rate, xavier, iterations, 1, gradient_check,
                   print_cost=print_cost)
    gc_model.gradient_check(L2, print_gradients=False)

    expected_result = {'J': 0.69314719280598658, 'grad_diff': 8.7674820454588602e-09}
    print('GC model result', gc_model.unit_test)
    print('GC model expected', expected_result)
    if gc_model.unit_test == expected_result:
        print("GC model unit test: OK!!!")
    else:
        print("GC model results don't match expected results. Please check!!!")
    return


if __name__ == "__main__":
    unit_test()
