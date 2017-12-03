# File name: NN
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
        gradient_parameters(dict): Dictionary with ndarray of gradients of weights and biases.



    '''

    def __init__(self, L, activations, unit_test=False):

        self.L = L
        self.layers = len(self.L)
        self.activations = activations
        self.parameters = {}
        self.gradient_parameters = {}
        self._update_count = 1

    def initialize_parameters(self, method='xavier', epsilon=0.01, beta1=0.0, beta2=0.0, seed=1):
        '''
        Initializes dictionary of parameters of weights and biases

        Args:
            method(str): Method of weights initialization, 'xavier','he' or nothing.
            epsilon(float): If xavier is false this will be used as the mean of the weights.
            beta1(float): Momentum beta1, if 0 then there is no momentum.
            beta2(float): RMSprop beta2, 0 if 0 then there is no rmsprop.
            seed(int): Ramdom number generator seed.

        '''

        np.random.seed(seed)

        for l in range(1, self.layers):
            if method == 'xavier':
                factor = np.sqrt(1 / self.L[l - 1])

            elif method == 'he':
                factor = np.sqrt(2 / self.L[l - 1])

            else:
                factor = epsilon
            self.parameters['W' + str(l)] = np.random.randn(self.L[l], self.L[l - 1]) * factor
            self.parameters['b' + str(l)] = np.zeros((self.L[l], 1))
            if beta1 != 0:
                self.gradient_parameters['dVW' + str(l)] = np.zeros((self.L[l], self.L[l - 1]))
                self.gradient_parameters['dVb' + str(l)] = np.zeros((self.L[l], 1))
            if beta2 != 0:
                self.gradient_parameters['dSW' + str(l)] = np.zeros((self.L[l], self.L[l - 1]))
                self.gradient_parameters['dSb' + str(l)] = np.zeros((self.L[l], 1))

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

        XT = X[:].T

        cache['A0'] = XT
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

        YT = Y[:].T
        m = AL.shape[1]

        cost = -np.multiply(YT, np.log(AL)) - np.multiply(1 - YT, np.log(1 - AL))
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
        YT = Y[:].T
        grads = {}

        m = AL.shape[1]

        AL[AL == 0] = 10e-8
        AL[AL == 1] = 1 - 10e-8
        dA = (-np.divide(YT, AL) + np.divide(1 - YT, 1 - AL)) / m
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

    def update_parameters(self, grads, learning_rate, beta1=0, beta2=0, epsilon=1e-8, correction=True):
        '''
        Update the weights and biases with the learning rate and the gradients.

        Args:
            grads(dict): Dictionary of gradients.
            learning_rate(float): learning rate.
            beta1(float): Momentum beta1, if 0 then there is no momentum.
            beta2(float): RMSprop beta2, 0 if 0 then there is no rmsprop.
            epsilon(float): Epsilon to get around dividing by 0 error.
            correction(bool): True for immediate warm (bias correction).

        '''

        for parameter_name in self.parameters:

            gradient = grads['d' + parameter_name]
            if beta1 != 0:
                self.gradient_parameters['dV' + parameter_name] = beta1 * self.gradient_parameters[
                    'dV' + parameter_name] + \
                                                                  (1 - beta1) * grads['d' + parameter_name]
                if correction:
                    gradient = self.gradient_parameters['dV' + parameter_name] / (1 - beta1 ** self._update_count)
                else:
                    gradient = self.gradient_parameters['dV' + parameter_name]
                    # print(gradient)
            if beta2 != 0:
                self.gradient_parameters['dS' + parameter_name] = beta2 * self.gradient_parameters[
                    'dS' + parameter_name] + (1 - beta2) * np.square(grads['d' + parameter_name])
                # print(self.gradient_parameters['dS' + parameter_name])
                if correction:
                    dS_corrected = self.gradient_parameters['dS' + parameter_name] / (1 - beta2 ** self._update_count)
                    # print(dS_corrected)
                    gradient = np.divide(gradient,
                                         np.sqrt(dS_corrected) + epsilon)

                else:

                    gradient = np.divide(gradient, np.sqrt(self.gradient_parameters['dS' + parameter_name]) + epsilon)

            self.parameters[parameter_name] = self.parameters[parameter_name] - learning_rate * gradient
        self._update_count += 1

    def fit(self, train_X, train_Y, lambd, keep_prob, learning_rate, xavier=True, num_epochs=1000, mini_batch_size=32,
            beta1=0, beta2=0, correction=False, seed=1, gradient_check=False, print_cost=False):
        '''
        Trains the model given a X and Y and learning paramaters and returns the final cross entropy loss.

        Args:
            train_X (ndarry): Samples as rows, features in columns.
            train_Y(ndarry): Labels in rows.
            lambd(float): If not None or 0, you will get L2 regularization with L2 penalty.
            keep_prob(float or array): Must be between 0.0 and 1.0. If array must match activations dimension.
            learning_rate(float): Learning rate.
            xavier(boolean): True for Xavier initialization otherwise random initialization.
            num_epochs(int): Number of epochs.
            mini_batch_size(int): Mini-batch size.
            beta1(float): Momentum beta1, if 0 then there is no momentum.
            beta2(float): RMSprop beta2, 0 if 0 then there is no rmsprop.
            correction(bool): True for immediate warm (bias correction).
            seed(int): Ramdom number generator seed.
            gradient_check(boolean): Switches off dropout to allow checking gradient with a numerical check.
            print_cost(boolean): True to print cost as you train.

        Returns:
            J (float): Cross Entroy loss for the given X, Y.
        '''
        np.random.seed(seed)
        assert len(self.L) == len(self.activations) + 1, 'L different from activations'
        if type(keep_prob) == list:
            assert len(keep_prob) == len(self.activations), 'keep_prob array must much activation dimension'
        self.initialize_parameters(xavier, seed=seed, beta1=beta1, beta2=beta2
                                   )

        # X = X[:].T

        for i in range(0, num_epochs):
            seed = seed + 1
            mini_batches = self.random_mini_batches(train_X, train_Y, mini_batch_size, seed)
            for X, Y in mini_batches:
                AL, cache = self.forward_propogation(X, keep_prob=keep_prob, gradient_check=gradient_check)

                J = self.compute_cost(AL, Y, lambd)

                grads = self.back_propogation(AL, Y, cache, lambd)

                self.update_parameters(grads, learning_rate, beta1, beta2, correction=correction)
            if i % 1000 == 0 and print_cost:
                print(i, J)

        print(J)

        return J

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

        AL, _ = self.forward_propogation(X, keep_prob=1.0, gradient_check=False)

        Y_pred = (AL.T > 0.5) * 1
        if Y.shape[0] != 0:
            correct = Y_pred == Y.reshape(-1, 1)

            correct = np.sum(correct)

            accuracy = float(correct) / float(m)
            print('Accuracy is %s%%' % (accuracy * 100))
        else:
            accuracy = None

        return Y_pred, accuracy

    def roll_params(self):
        '''
        Rolls the parameters into a single vector of all parameters.
        Returns:
            vector_parameters[ndarray]: Single column array of parameters.

        '''

        vector_parameters = np.array([])
        for l in range(1, self.layers):
            vector_parameters = np.concatenate(
                [vector_parameters, self.parameters['W' + str(l)].flatten(), self.parameters['b' + str(l)].flatten()])

        return vector_parameters

    def roll_grads(self, grads):
        '''
        Rolls the gradients into a single vector of all gradients.
        Returns:
            vector_grads[ndarray]: Single column array of gradients.

        '''
        vector_grads = np.array([])
        for l in range(1, self.layers):
            vector_grads = np.concatenate(
                [vector_grads, grads['dW' + str(l)].flatten(), grads['db' + str(l)].flatten()])

        return vector_grads

    def unroll_params(self, vector_parameters):
        '''
        Unrolls a vector of parameters into a dictionary.
        Returns:
            parameters[dict]: Dictionary of all parameters.

        '''
        parameters = {}
        i = 0
        for l in range(1, self.layers):
            parameters['W' + str(l)] = vector_parameters[i:i + self.L[l - 1] * self.L[l]].reshape(self.L[l],
                                                                                                  self.L[l - 1])
            i = i + self.L[l - 1] * self.L[l]
            parameters['b' + str(l)] = vector_parameters[i:i + self.L[l]].reshape(self.L[l], 1)
            i = i + self.L[l]

        return parameters

    def unroll_grads(self, vector_grads):
        '''
        Unrolls a vector of gradients into a dictionary.
        Returns:
            grads[dict]: Dictionary of all gradients.

        '''
        grads = {}
        i = 0
        for l in range(1, self.layers):
            grads['dW' + str(l)] = vector_grads[i:i + self.L[l - 1] * self.L[l]].reshape(self.L[l], self.L[l - 1])
            i = i + self.L[l - 1] * self.L[l]
            grads['db' + str(l)] = vector_grads[i:i + self.L[l]].reshape(self.L[l], 1)
            i = i + self.L[l]

        return grads

    def gradient_check(self, X, Y, lambd, epsilon=1e-7, print_gradients=False):
        '''
        Performs the gradient check between the backprop actual gradient and numerical gradient.
        Args:
            X (ndarry): Samples as rows, features in columns.
            Y(ndarry): Labels in rows.
            lambd(float): If not None or 0, you will get L2 regularization with L2 penalty.
            epsilon(float): This is the epsilon used for derivative calculation.
            print_gradients(boolean): Set to True if you want to see the actual and numerical gradients for debugging.

        Returns:
            total_difference(float): The total difference between the numerical and actual gradients as a relative 2-norm.

        '''

        AL, cache = self.forward_propogation(X, keep_prob=1., gradient_check=True)

        grads = self.back_propogation(AL, Y, cache, lambd)
        old_parameters = self.parameters.copy()

        vector_parameters = self.roll_params()
        numerical_grads = []
        p_index = len(vector_parameters)
        for i in range(0, p_index):
            vector_params_plus = vector_parameters.copy()
            vector_params_plus[i] = vector_params_plus[i] + epsilon
            self.parameters = self.unroll_params(vector_params_plus)

            AL, _ = self.forward_propogation(X, keep_prob=1.0, gradient_check=True)
            J_plus = self.compute_cost(AL, Y, lambd)

            vector_params_minus = vector_parameters.copy()
            vector_params_minus[i] = vector_params_minus[i] - epsilon
            self.parameters = self.unroll_params(vector_params_minus)

            AL, _ = self.forward_propogation(X, keep_prob=1.0, gradient_check=True)
            J_minus = self.compute_cost(AL, Y, lambd)

            numerical_grads.append((J_plus - J_minus) / 2. / epsilon)

        numerical_grads = np.array(numerical_grads)
        num_grads = self.unroll_grads(numerical_grads)

        actual_grads = self.roll_grads(grads)
        differences = numerical_grads - actual_grads

        total_difference = np.linalg.norm(differences) / (
            np.linalg.norm(numerical_grads) + np.linalg.norm(actual_grads))

        print('2-norm relative difference of gradients: ', total_difference)

        if print_gradients:
            print('numerical gradient:\n', sorted(num_grads.items()), '\n')
            print('actual gradient:\n', sorted(grads.items()))
        self.parameters = old_parameters

        if total_difference < 1e-7:
            print('Gradient calculation in backprop looks to be ok. Repeat for after a number of iterations.')
        else:
            print('Please check your gradient calculation in backprop.')
        return total_difference

    def random_mini_batches(self, X, Y, mini_batch_size, seed):
        '''
        Takes X, Y and splits into mini-batches.

        Args:
            X (ndarry): Samples as rows, features in columns.
            Y(ndarry): Labels in rows.
            mini_batch_size: Mini-batch size.
            seed(int): Ramdom number generator seed.

        Returns:
            mini_batches(list): List of pairs of X,Y mini-batches.
        '''
        if mini_batch_size:
            np.random.seed(seed)
            m = X.shape[0]

            indices = list(np.random.permutation(m))

            num_splits = m // int(mini_batch_size)
            mini_batches = []
            if num_splits > 0:
                for i in range(0, num_splits):
                    split = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
                    X_batch = X[split, :]
                    Y_batch = Y[split]
                    mini_batches.append((X_batch, Y_batch))

            split = indices[num_splits * mini_batch_size:]
            X_batch = X[split, :]
            Y_batch = Y[split]
            mini_batches.append((X_batch, Y_batch))
            return mini_batches
        else:
            return [(X, Y)]
