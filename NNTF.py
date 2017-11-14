
import tensorflow as tf

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

     '''
    def __init__(self, L, activations):


        self.X = tf.placeholder(shape=[None, L[0]], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.L = L
        self.activations = activations
        self.saver_path = "/tmp/model.ckpt"

    def nn(self):
        '''
        Create the neural network up to the ZL which will be before the final activation.
        '''

        lin_activations = {}
        layers = len(self.activations) - 1
        lin_activations[0] = self.X

        for l in range(0, layers):
            lin_activations[l + 1] = tf.contrib.layers.fully_connected(
                activation_fn=activation_functions[self.activations[l]],
                biases_initializer=tf.zeros_initializer,
                inputs=lin_activations[l], num_outputs=self.L[l + 1],
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )

        output_layer = tf.contrib.layers.fully_connected(activation_fn=None,
                                                         biases_initializer=tf.zeros_initializer,
                                                         inputs=lin_activations[layers],
                                                         num_outputs=self.L[layers + 1],
                                                         weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                         )
        return output_layer

    def fit(self, X, Y, L2, keep_prob, learning_rate, xavier=True, iterations=1000, gradient_check=False,
            print_cost=False):
        '''
        Trains the model given a X and Y and learning paramaters.

        Args:
            X (ndarry): Samples as columns, features in rows.
            Y(ndarry): Labels.
            L2(float): If not None or 0, you will get L2 regularization with L2 penalty.
            keep_prob(float): If less than 1.0, dropout regularization will be implemented.
            learning_rate(float): Learning rate.
            xavier(boolean): True for Xavier initialization otherwise random initialization.
            iterations(int): Number of iterations.
            gradient_check(boolean): Switches off dropout to allow checking gradient with a numerical check.
            print_cost(boolean): True to print cost as you train.

        Returns:
            tuple: parameters(dict): Learned parameters.
            grad(dict): Learned gradients.
        '''

        self.output_layer = self.nn()

        J = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.output_layer))

        self.predictions = tf.cast(tf.greater(self.output_layer, tf.constant(0., dtype=tf.float32)), dtype=tf.int32)
        Y_int = tf.cast(self.Y, dtype=tf.int32)
        self.accuracy = tf.contrib.metrics.accuracy(labels=Y_int, predictions=self.predictions)

        init_gl = tf.global_variables_initializer()
        init_lc = tf.local_variables_initializer()

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(J)
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run([init_gl, init_lc])

        for i in range(0, iterations):
            if i % 100 == 99:
                print(i, ':', sess.run(J, feed_dict={self.X: X, self.Y: Y}))
            sess.run(optimizer, feed_dict={self.X: X, self.Y: Y})

        saver.save(sess, self.saver_path)
        sess.close()

    def predict(self, X, Y):
        '''
        Returns the preidctions of the model and the accuracy score.

        Args:
            X (ndarry): Samples as columns, features in rows.
            Y(ndarry): Labels.


        Returns:
            tuple: Y_pred(ndarray): Predicted labels.
            accuracy(float): Accuracy score.
        '''

        saver = tf.train.Saver()

        sess = tf.Session()

        saver.restore(sess, self.saver_path)

        Y_pred, accuracy = sess.run([self.predictions, self.accuracy], feed_dict={self.X: X, self.Y: Y})

        sess.close()

        print('Accuracy is %s%%' % (accuracy * 100))
        return Y_pred, accuracy
