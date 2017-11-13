import numpy as np
from nn_model import run_model, predict
class data(object):
    '''
    Class for dataset objects.
    '''
   
    
    def __init__(self):
        '''
        Args:
            train_path(str): Training data file path.
            test_path(str): Testing data file path.

        '''

        self.train_X_orig, self.train_Y, self.test_X_orig, self.test_Y,self.classes  = self.load_data(train_path, test_path)
        self.train_X, self.test_X = self.transform_data()
        return
    
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
        Y= self.test_Y
        Y_pred = self.test_Y_pred
        classes= self.classes
        
        wrong = Y != Y_pred
        wrong = np.squeeze(wrong)
        
        self.errors = wrong
    
    def train(self,L, activations, L2, keep_prob, learning_rate, xavier, iterations, gradient_check, print_cost=True):
        '''
        Trains the neural network using the training data.

        Args:
            L(list): List of nodes in each of the layers including the input and output lauer.
            activations(list):  List of activations in the different layers {'relu', 'sigmoid', 'leaky-relu'}.
            L2(float): If not None or 0, you will get L2 regularization with L2 penalty.
            keep_prob(float): If less than 1.0, dropout regularization will be implemented.
            learning_rate(float): Learning rate.
            xavier(boolean): True for Xavier initialization otherwise random initialization.
            iterations(int): Number of iterations.
            gradient_check(boolean): Switches off dropout regularization to allow checking gradient with a numerical check.
            print_cost(boolean): True to print cost as you train.

        '''
        self.activations = activations
        X = self.train_X
        Y = self.train_Y
        self.parameters, grads = run_model(X, Y, L2, keep_prob, learning_rate,L, activations, xavier = xavier, iterations=iterations, gradient_check = gradient_check, print_cost = print_cost)

        return
    
    def predict_train(self):
        '''
        Calculates the prediction and accuracy from the training data.

        '''
        X = self.train_X
        Y = self.train_Y
        parameters = self.parameters
        activations = self.activations
    
        self.train_Y_pred, self.train_accuracy = predict(X,Y, parameters, activations)

    def predict_test(self):
        '''
        Calculates the prediction and accuracy from the test data.


        '''
        X = self.test_X
        Y = self.test_Y
        parameters = self.parameters
        activations = self.activations
    
        self.test_Y_pred, self.test_accuracy = predict(X,Y, parameters, activations)
        

