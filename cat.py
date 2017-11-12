from data import data
import numpy as np
import h5py
import matplotlib.pyplot as plt
np.random.seed(1)
class cat_data(data):
    # import PIL
    def load_data(self, train_path, test_path):
        train_dataset = h5py.File(train_path, 'r')
        test_dataset = h5py.File(test_path,'r')
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

        train_set_y_orig = train_set_y_orig.reshape(-1,train_set_y_orig.shape[0])
        test_set_y_orig = test_set_y_orig.reshape(-1,test_set_y_orig.shape[0])
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def transform_data(self):
        m = self.train_X_orig.shape[0]
        train_set_x_orig = self.train_X_orig.reshape(m, -1).T/255.
        # train_set_x_orig = train_set_x_orig[:100,:]
        m_test = self.test_X_orig.shape[0]
        test_set_x_orig = self.test_X_orig.reshape(m_test, -1).T/255.
        return train_set_x_orig, test_set_x_orig
    
    def show_errors(self, num_errors = 5):
        super(cat_data,self).show_errors()

        show_num_errors = min(num_errors, np.sum(self.errors*1))

        classification = self.test_Y[:,self.errors]
        prediction = self.test_Y_pred[:,self.errors]
        images = self.test_X_orig[self.errors]


        for i in range(0, show_num_errors):
            self.show_data(i, 3, images, classification)
            print('Prediction is %s' % self.classes[prediction[0,i]])


    def show_data(self, index, size = 6, X = np.array([]), Y = np.array([])):
        
        if X.shape[0] == 0:
            X = self.train_X_orig
        if Y.shape[0] == 0:
            Y = self.train_Y
        classes = self.classes
        
        plt.rcParams['figure.figsize'] = (size,size)
        plt.imshow(X[index, :])
        plt.show()

        classification = classes[Y[0, index]]
        print('This is a %s' % classification)
        

def unit_test():
    # Test number one should print This is a cat Accuracy is 98.5645933014% Accuracy is 80.0%
    cat_dataset = cat_data('./dataset/train_catvnoncat.h5', './dataset/test_catvnoncat.h5')
    cat_dataset.show_data(2)
    L = [12288,20,7,5,1]
    activations = ['relu','relu','relu','sigmoid']
    L2 = 0
    keep_prob = 1.
    learning_rate = 0.0075
    iterations = 2500
    gradient_check= True
    print_cost = False
    xavier = True


    cat_dataset.learn(L, activations, L2, keep_prob, learning_rate, xavier, iterations, gradient_check, print_cost=print_cost)
    cat_dataset.predict_train()
    cat_dataset.predict_test()
    cat_dataset.show_errors()

if __name__ == "__main__":
    unit_test()
