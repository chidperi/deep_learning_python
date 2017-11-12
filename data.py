import numpy as np
from nn_model import run_model, predict
class data(object):
   
    
    def __init__(self, train_path, test_path):

        self.train_X_orig, self.train_Y, self.test_X_orig, self.test_Y,self.classes  = self.load_data(train_path, test_path)
        self.train_X, self.test_X = self.transform_data()
        return
    
    def load_data(self, train_path, test_path):
        return
    
#     def show_data(self):
#         return
    
    def transform_data(self):
        
        return
    
    def show_errors(self):
        X = self.test_X_orig
        Y= self.test_Y
        Y_pred = self.test_Y_pred
        classes= self.classes
        
        wrong = Y != Y_pred
        wrong = np.squeeze(wrong)
        
        self.errors = wrong
    
    def learn(self,L, activations, L2, keep_prob, learning_rate, xavier, iterations, gradient_check, print_cost=True):
        
        self.activations = activations
        X = self.train_X
        Y = self.train_Y
        self.parameters, grads = run_model(X, Y, L2, keep_prob, learning_rate,L, activations, xavier = xavier, iterations=iterations, gradient_check = gradient_check, print_cost = print_cost)

        return
    
    def predict_train(self):
        X = self.train_X
        Y = self.train_Y
        parameters = self.parameters
        activations = self.activations
    
        self.train_Y_pred, self.train_accuracy = predict(X,Y, parameters, activations)

    def predict_test(self):
        X = self.test_X
        Y = self.test_Y
        parameters = self.parameters
        activations = self.activations
    
        self.test_Y_pred, self.test_accuracy = predict(X,Y, parameters, activations)
        

