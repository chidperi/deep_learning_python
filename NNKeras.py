# File name: NNKeras
# Copyright 2017 Chidambaram Periakaruppan 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the 
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

np.random.seed(1337)  # for reproducibility
import tensorflow as tf

tf.set_random_seed(2)
from keras.layers import Input, Dense, Activation, Dropout
from keras.regularizers import l2
from keras.models import Model
from keras.initializers import lecun_normal
from keras.optimizers import SGD


class NNKeras(object):
    '''Creates the Keras neural network.

     Args:
         L(list): List of nodes in each of the layers including the input and output lauer.
         activations(list): List of activations in the different layers {‘relu’, ‘sigmoid’, ‘leaky-relu’}.

     Attributes

         L(list): List of nodes in each of the layers including the input and output lauer.
         activations(list): List of activations in the different layers {‘relu’, ‘sigmoid’, ‘leaky-relu’}.
         model(keras.Model): keras.Model object.
         keep_prob(float or list): If below 1.0, dropout regularization will be implemented.

     '''

    def __init__(self, L, activations):

        assert len(L) == len(activations) + 1, 'L different from activations'

        self.L = L
        self.activations = activations
        self.model = None
        self.keep_prob = []

    def fit(self, X, Y, lambd, keep_prob, learning_rate, xavier=True, iterations=1000, seed=1, gradient_check=False,
            print_cost=False):
        '''
        Trains the model given a X and Y and learning paramaters.

        Args:
            X (ndarry): Samples as columns, features in rows.
            Y(ndarry): Labels.
            lambd(float): If not None or 0, you will get L2 regularization with L2 penalty.
            keep_prob(float): If less than 1.0, dropout regularization will be implemented.
            learning_rate(float): Learning rate.
            xavier(boolean): True for Xavier initialization otherwise random initialization.
            iterations(int): Number of iterations.
            seed(int): Ramdom number generator seed.
            gradient_check(boolean): Switches off dropout to allow checking gradient with a numerical check.
            print_cost(boolean): True to print cost as you train.

        '''

        if type(keep_prob) == list:
            assert len(keep_prob) == len(self.activations), 'keep_prob array must much activation dimension'''

        if type(keep_prob) != list:
            self.keep_prob = np.ones(len(self.activations)) * (keep_prob)
        else:
            self.keep_prob = [x for x in keep_prob]

        inputs = Input(shape=[self.L[0], ])
        x = inputs
        for i in range(1, len(self.L)):
            name = str(i)
            print(name, self.activations[i - 1], self.keep_prob[i - 1])
            x = Dense(self.L[i], name=name + 'Z', kernel_initializer=lecun_normal(seed=seed),
                      kernel_regularizer=l2(lambd))(x)
            x = Activation(self.activations[i - 1], name=name + 'A')(x)
            x = Dropout(1 - self.keep_prob[i - 1], seed=seed, name=name + 'D')(x)

        output_layer = x

        self.model = Model(input=[inputs], output=[output_layer])
        self.model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(x=X, y=Y, verbose=0, epochs=iterations, batch_size=X.shape[0], shuffle=False)

    def predict(self, X, Y=np.array([])):
        '''
        Returns the preidctions of the model and the accuracy score.

        Args:
            X (ndarry): Samples as columns, features in rows.
            Y(ndarry): Labels. If empty array there will be no accuracy score.


        Returns:
            tuple: Y_pred(ndarray): Predicted labels.
            accuracy(float): Accuracy score.
        '''

        no_drop = np.ones(len(self.activations))
        Y_pred = self.model.predict(x=X, batch_size=X.shape[0])
        Y_pred = Y_pred > 0.5
        Y_pred = Y_pred * 1

        accuracy = None
        if Y.shape[0] != 0:
            accuracy = self.model.evaluate(x=X, y=Y, batch_size=X.shape[0])
            print('Accuracy is %s%%' % (accuracy * 100))

        return Y_pred, accuracy
