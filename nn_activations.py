import numpy as np

def sigmoid(x):
    '''
        input: x is a vector or scaler
        output: y is the sigmoid activation of y
                deriv is the derivative at pt y
    '''
    y = 1./(1+np.exp(-x))
    deriv = np.multiply(y, 1.-y)
    
    return y, deriv
    
def relu(x):
    '''
        input: x is a vector or scaler
        output: y is the relu activation of y
                deriv is the derivative at pt y
    '''
    y = np.maximum(0, x)
    deriv = (x>0)*1
    
    return y, deriv 

def leaky_relu(x, leak = 0.1):
    '''
        input: x is a vector or scaler
        output: y is the leaky relu activation of y
                deriv is the derivative at pt y
    '''
    y = np.maximum(leak*x, x)
    deriv = (x<0) * leak
    deriv = deriv + (x>0) * 1
    
    return y, deriv 


        
            

activation_functions = {'relu': relu, 'leaky_relu': leaky_relu, 'sigmoid':sigmoid}

