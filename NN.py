# File name: test
# Copyright 2017 Chidambaram Periakaruppan
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from NNActivations import activation_functions


class NN:
    '''Creates the NN instance which is a python neural network.

    Args:
        L(list): List of nodes in each of the layers including the input and output lauer.
        activations(list): List of activations in the different layers {‘relu’, ‘sigmoid’, ‘leaky-relu’}.

    Attributes
        L(list): List of nodes in each of the layers including the input and output lauer.
        layers(int): Number of layers.
        activations(list): List of activations in the different layers {‘relu’, ‘sigmoid’, ‘leaky-relu’}.
        parameters(dict): Dictionary with ndarray of weights and biases.
        keep_prob(array-like): List of keep_probs for the different layers.

    '''
    def __init__(self, L, activations):

        self.L = L
        self.layers = len(self.L)
        self.activations = activations
        self.parameters = {}

    def initialize_parameters(self, xavier=True, epsilon=0.01, seed=1):
        '''
        Initializes dictionary of parameters of weights and biases

        Args:
            xavier(boolean): True for Xavier initialization otherwise random initialization.
            epsilon(float): If xavier is false this will be used as the mean of the weights.
            seed(int): Ramdom number generator seed.

        '''

        np.random.seed(seed)

        for l in range(1, self.layers):
            self.parameters['W' + str(l)] = np.random.randn(self.L[l], self.L[l - 1]) * \
                                            (1 / np.sqrt(self.L[l - 1]) if xavier else epsilon)
            self.parameters['b' + str(l)] = np.zeros((self.L[l], 1))

    def forward_propogation(self, X, keep_prob=1.0, gradient_check=False):
        '''
        Return the final activation and cache after forward proppogation.

        Args:
            X (ndarry): Samples as columns, features in rows.
            keep_prob(float): If less than 1.0, dropout regularization will be implemented.
            gradient_check(boolean): Switches off dropout to allow checking gradient with a numerical check.
        Returns:
            tuple:  A (ndarray), Final activation for each sample as array of floats,
            cache (dict), dictionary of the Z, A and derivs for each layer.

        '''
        np.random.seed(1)
        cache = {}

        if gradient_check:
            keep_prob = 1.0

        if type(keep_prob) != list:
            keep_prob = np.ones(len(self.activations)) * keep_prob



        cache['A0'] = X
        for l in range(1, self.layers):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            G = activation_functions[self.activations[l - 1]]

            Z = np.matmul(W, cache['A' + str(l - 1)]) + b
            A, deriv = G(Z)

            keep = np.random.rand(A.shape[0], A.shape[1])
            keep = keep <= keep_prob[l - 1]
            keep = keep * 1. / keep_prob[l - 1]
            A = np.multiply(A, keep)

            deriv = np.multiply(deriv, keep)

            # cache['Z' + str(l)] = Z
            # cache['D'+str(l)] = keep
            cache['A' + str(l)] = A
            cache['deriv' + str(l)] = deriv

        return A, cache

    def L2_cost(self):
        '''
        Returns the L2 norm of all the weights.

        Returns:
            float: L2 norm of weights.

        '''
        L2 = 0.

        for parameter_name, parameter in self.parameters.items():

            if 'W' in parameter_name:

                L2 += np.sum(np.square(parameter))
        return L2

    def compute_cost(self, AL, Y, lambd):
        '''
        Computes the cost cross entropy cost of the precictions.

        Args:
            AL(ndarray): Final activations (logits).
            Y(ndarry): Labels.
            lambd(float): if not None or 0, you will get L2 regularization with L2 penalty.

        Returns:
            float: cost.
        '''

        m = AL.shape[1]
        cost = -np.multiply(Y, np.log(AL)) - np.multiply(1 - Y, np.log(1 - AL))
        cost = np.nansum(cost)
        if lambd:
            cost += lambd * self.L2_cost() / 2.

        cost = cost / m

        cost = np.squeeze(cost)
        return cost

    def back_propogation(self, AL, Y, cache, L2):
        '''
        Returns the gradients after backpropogation.

        Args:
            AL(ndarry): Final activations (logits).
            Y(ndarry): Labels.
            cache (dict): Dictionary of the Z, A and derivs for each layer.
            L2(float): If not None or 0, you will get L2 regularization with L2 penalty.

        Returns:
            dict: Dictionary of gradients.
        '''
        grads = {}

        m = AL.shape[1]

        AL[AL == 0] = 10e-8
        AL[AL == 1] = 1 - 10e-8
        dA = (-np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)) / m
        for l in range(self.layers - 1, 0, -1):
            dAdZ = cache['deriv' + str(l)]

            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            A_prev = cache['A' + str(l - 1)]

            dZ = np.multiply(dA, dAdZ)


            dW = np.matmul(dZ, A_prev.T) + L2 * W / m
            db = np.sum(dZ, axis=1, keepdims=True)

            dA_prev = np.matmul(W.T, dZ)

            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db
            dA = dA_prev

        return grads

    def update_parameters(self, grads, learning_rate):
        '''
        Update the weights and biases with the learning rate and the gradients.

        Args:
            grads(dict): Dictionary of gradients.
            learning_rate(float): learning rate.

        '''
        for parameter_name in self.parameters:
            self.parameters[parameter_name] = self.parameters[parameter_name] - learning_rate * grads[
                'd' + parameter_name]

    def fit(self, X, Y, lambd, keep_prob, learning_rate, xavier=True, iterations=1000, seed=1, gradient_check=False,
            print_cost=False):
        '''
        Trains the model given a X and Y and learning paramaters.

        Args:
            X (ndarry): Samples as columns, features in rows.
            Y(ndarry): Labels.
            lambd(float): If not None or 0, you will get L2 regularization with L2 penalty.
            keep_prob(float or array): Must be between 0.0 and 1.0. If array must match activations dimension.
            learning_rate(float): Learning rate.
            xavier(boolean): True for Xavier initialization otherwise random initialization.
            iterations(int): Number of iterations.
            seed(int): Ramdom number generator seed.
            gradient_check(boolean): Switches off dropout to allow checking gradient with a numerical check.
            print_cost(boolean): True to print cost as you train.

        Returns:
            tuple: parameters(dict): Learned parameters.
            grad(dict): Learned gradients.
        '''
        np.random.seed(seed)
        assert len(self.L) == len(self.activations) + 1, 'L different from activations'
        if type(keep_prob) == list:
            assert len(keep_prob) == len(self.activations), 'keep_prob array must much activation dimension'
        self.initialize_parameters(xavier, seed=seed)


        X = X[:].T

        for i in range(0, iterations):

            AL, cache = self.forward_propogation(X, keep_prob=keep_prob, gradient_check=gradient_check)

            J = self.compute_cost(AL, Y.T, lambd)
            grads = self.back_propogation(AL, Y.T, cache, lambd)


            self.update_parameters(grads, learning_rate)
            if i % 10000 == 0 and print_cost:
                print(i, J)
                # print(i, J)

    def predict(self, X, Y=np.array([])):
        '''
        Returns the preidctions of the model and the accuracy score.

        Args:
            X (ndarry): Samples as columns, features in rows.
            Y(ndarry): Labels. If empty array there will be no accuracy score.


        Returns:
            tuple: Y_pred(ndarray): Predicted labels.
            accuracy(float): Accuracy score if Y array is not empty.
        '''

        m = Y.shape[0]
        AL, _ = self.forward_propogation(X.T, keep_prob=1.0, gradient_check=False)

        Y_pred = (AL.T > 0.5) * 1

        if Y.shape[0] != 0:
            correct = Y_pred == Y
            correct = np.sum(correct)
            accuracy = float(correct) / float(m)
            print('Accuracy is %s%%' % (accuracy * 100))
        else:
            accuracy = None

        return Y_pred, accuracy
