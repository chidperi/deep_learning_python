from Model import Model
import numpy as np
import h5py
import matplotlib.pyplot as plt
from NN import NN
from NNTF import NNTF

np.random.seed(1)


class CatModel(Model):
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
        train_dataset = h5py.File(train_path, 'r')
        test_dataset = h5py.File(test_path, 'r')
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

        train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0], -1)
        test_set_y_orig = test_set_y_orig.reshape(test_set_y_orig.shape[0], -1)
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def transform_data(self):
        '''
        Transforms the original data so that they are normalized and samples are in columns and features are in rows.
        Returns:
            tuple: train_set_x_orig(ndarray): Transformed training data.
            test_set_x_orig(ndarray): Transformed test data.

        '''
        m = self.train_X_orig.shape[0]
        train_set_x_orig = self.train_X_orig.reshape(m, -1) / 255.
        # train_set_x_orig = train_set_x_orig[:100,:]
        m_test = self.test_X_orig.shape[0]
        test_set_x_orig = self.test_X_orig.reshape(m_test, -1) / 255.
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

    cat_model = CatModel('./dataset/train_catvnoncat.h5', './dataset/test_catvnoncat.h5')
    # cat_model.show_data(2)
    L = [12288, 20, 7, 5, 1]
    activations = ['relu', 'relu', 'relu', 'sigmoid']
    L2 = 0
    keep_prob = 1.
    learning_rate = 0.0075
    iterations = 2500
    gradient_check = True
    print_cost = True
    xavier = True

    cat_model.train(NNTF, L,activations, L2, keep_prob, learning_rate, xavier, iterations, gradient_check,
                       print_cost=print_cost)
    cat_model.predict_train()
    cat_model.predict_test()
    cat_model.show_errors()


if __name__ == "__main__":
    unit_test()
