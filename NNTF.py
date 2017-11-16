# File name: test
# Copyright 2017 Chidambaram Periakaruppan
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import tensorflow as tf
import numpy as np

activation_functions = {'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid}


class NNTF(object):
    '''Creates the TensorFlow neural network.

     Args:
         L(list): List of nodes in each of the layers including the input and output lauer.
         activations(list): List of activations in the different layers {‘relu’, ‘sigmoid’, ‘leaky-relu’}.

     Attributes
         X(tf.placeholder): Placeholder for X.
         Y(tf.placeholder): Placeholder for Y.
         L(list): List of nodes in each of the layers including the input and output lauer.
         activations(list): List of activations in the different layers {‘relu’, ‘sigmoid’, ‘leaky-relu’}.
         saver_path(str): Path to save the variables after training.
         keep_prob(float or list): If below 1.0, dropout regularization will be implemented.

     '''
    def __init__(self, L, activations):

        assert len(L) == len(activations) + 1, 'L different from activations'

        tf.reset_default_graph()
        self.X = tf.placeholder(shape=[None, L[0]], dtype=tf.float64)
        self.Y = tf.placeholder(shape=[None, 1], dtype=tf.float64)
        self.L = L
        self.activations = activations
        self.saver_path = "/tmp/model.ckpt"
        self.keep_prob = tf.placeholder(shape=[len(self.activations)], dtype=tf.float64)

    def nn(self, L2, seed=1):
        '''
        Create the neural network up to the ZL which will be before the final activation.
        '''

        lin_activations = {}
        layers = len(self.activations) - 1
        lin_activations['0d'] = self.X




        for l in range(0, layers):
            lin_activations[str(l + 1)] = tf.contrib.layers.fully_connected(
                activation_fn=activation_functions[self.activations[l]],
                biases_initializer=tf.zeros_initializer,
                inputs=lin_activations[str(l) + 'd'], num_outputs=self.L[l + 1],
                weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                weights_regularizer=L2
                )

            lin_activations[str(l + 1) + 'd'] = tf.nn.dropout(x=lin_activations[str(l + 1)],
                                                              keep_prob=self.keep_prob[l],
                                                              seed=seed
                                                              )


        output_layer = tf.contrib.layers.fully_connected(activation_fn=None,
                                                         biases_initializer=tf.zeros_initializer,
                                                         inputs=lin_activations[str(layers) + 'd'],
                                                         num_outputs=self.L[layers + 1],
                                                         weights_initializer=tf.contrib.layers.xavier_initializer(
                                                             seed=seed),
                                                         weights_regularizer=L2

                                                         )
        return output_layer

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
            keep_prob = np.ones(len(self.activations)) * (keep_prob)
        else:
            keep_prob = [x for x in keep_prob]

        lambd_1 = tf.constant(lambd, dtype=tf.float64)

        L2 = tf.contrib.layers.l2_regularizer(lambd_1)

        no_drop = np.ones(len(self.activations))

        tf.set_random_seed(seed)

        self.output_layer = self.nn(L2)

        L2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) / tf.cast(
            tf.shape(self.output_layer)[0], dtype=tf.float64)

        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.output_layer))

        J = cost + L2_loss
        # 0: 0.694912471704
        # 10000: 0.269567532447
        self.predictions = tf.cast(tf.greater(self.output_layer, tf.constant(0., dtype=tf.float64)), dtype=tf.int64)
        Y_int = tf.cast(self.Y, dtype=tf.int64)
        self.accuracy = tf.contrib.metrics.accuracy(labels=Y_int, predictions=self.predictions)

        init_gl = tf.global_variables_initializer()
        init_lc = tf.local_variables_initializer()

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(J)
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run([init_gl, init_lc])

        for i in range(0, iterations):

            _, total_cost = sess.run([optimizer, J], feed_dict={self.X: X, self.Y: Y, self.keep_prob: keep_prob})
            if i % 10000 == 0:
                print(i, ':', total_cost)

        saver.save(sess, self.saver_path)
        sess.close()

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
        saver = tf.train.Saver()

        sess = tf.Session()

        saver.restore(sess, self.saver_path)
        if Y.shape[0] != 0:

            Y_pred, accuracy = sess.run([self.predictions, self.accuracy],
                                        feed_dict={self.X: X, self.Y: Y, self.keep_prob: no_drop})
            print('Accuracy is %s%%' % (accuracy * 100))
        else:
            Y_pred = sess.run(self.predictions,
                              feed_dict={self.X: X, self.keep_prob: no_drop})
            accuracy = None
        return Y_pred, accuracy
