import numpy as np
from nn_activations import activation_functions
def initialize_parameters(L, xavier = True, epsilon = 0.01):
    '''
        inputs: L a list of hiddend units in all layers from 0 to L
        outputs: parameters, a dictionary of W, b for each layer of L.
    '''
    
    np.random.seed(1)
    parameters = {}
    layers = len(L)
    for l in range(1,layers):
        parameters['W' + str(l)] = np.random.randn(L[l], L[l-1])* (1/np.sqrt(L[l-1]) if xavier else epsilon)
        parameters['b' + str(l)] = np.zeros((L[l], 1))
        
    return parameters
        




def forward_propogation(X, parameters, activations, keep_prob = 1.0, gradient_check = False):
    '''
        input: X is the input or A_0 with sample as columns
        parameters: W and b for each layer in a dictionary
        L: list of layer's hidden units
        activations: activation functions for each layer
        
        output: AL which is the final activation value
    '''
    cache = {}
    
    if gradient_check:
        keep_prob = 1.0
    
    layers = len(activations)
    cache['A0'] = X
    for l in range(1,layers+1):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        G = activation_functions[activations[l-1]]

            
        Z = np.matmul(W, cache['A' + str(l-1)]) + b

        if keep_prob < 1:
            n_b_x, n_b_y = b.shape
            keep = np.random.rand(n_b_x,n_b_y)
            keep = keep < keep_prob
            Z = np.multiply(Z, keep) / keep_prob
            
        A,deriv = G(Z)
        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A
        cache['deriv' + str(l)] = deriv
        
    return A, cache

def L2_cost(parameters):
    '''
        input: parameters
        output: L2 which is the L2 cost of 
    '''
    L2 = 0.
    for parameter_name, parameter in parameters.items():
        if 'W' in parameter_name:
           L2 += np.sum(np.square(parameter))
        return L2
def compute_cost(AL, Y, parameters, L2):

    m = AL.shape[1]
    cost = -np.matmul(Y, np.log(AL.T)) - np.matmul(1-Y, np.log(1-AL.T))
    if L2:
        cost += L2 * L2_cost(parameters) / 2.
    
    cost = cost/m

    cost = np.squeeze(cost)
    return cost

def back_propogation(AL,Y, parameters,activations, cache, L2):
    grads = {}
    layers = len(activations)
    m = AL.shape[1]
    dA =(-np.divide(Y,AL) + np.divide(1-Y, 1-AL))/m
    for l in range(layers,0,-1):

        dAdZ= cache['deriv' + str(l)]
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A_prev = cache['A' + str(l-1)]

        dZ = np.multiply(dA,dAdZ)

        dW = np.matmul(dZ, A_prev.T) + L2 * W / m
        db = np.sum(dZ, axis = 1, keepdims=True)
        dA_prev = np.matmul(W.T,dZ)

        grads['dW'+str(l)]= dW
        grads['db'+str(l)]= db
        dA= dA_prev
        
    return grads
        
    
def update_parameters(parameters, grads, learning_rate):
    for parameter_name in parameters:
        parameters[parameter_name] =parameters[parameter_name]- learning_rate * grads['d' + parameter_name]

    return parameters
    
    
def run_model(X, Y, L2, keep_prob, learning_rate,L, activations, xavier = True, iterations=1000, gradient_check = False, print_cost = False):

    np.random.seed(1)
    assert len(L) == len(activations)+1, 'L different from activations'
    parameters = initialize_parameters(L, xavier)
#     print parameters

    for i in range(0, iterations):
        AL,cache = forward_propogation(X,parameters,activations, keep_prob= keep_prob, gradient_check=gradient_check)

        J = compute_cost(AL, Y, parameters,L2 )
        grads = back_propogation(AL,Y, parameters, activations, cache, L2)

        parameters = update_parameters(parameters,grads, learning_rate)
        if i%100 == 0 and print_cost:
            print(J)
    return parameters, grads
        





def predict(X,Y, parameters, activations):
    '''
        input: X is the input or A_0 with sample as columns
        parameters: W and b for each layer in a dictionary
        L: list of layer's hidden units
        activations: activation functions for each layer
        
        output: AL which is the final activation value
    '''
    m = Y.shape[1]
    AL, _ = forward_propogation(X, parameters, activations, keep_prob = 1.0, gradient_check = False)
    
    Y_pred = (AL>0.5)*1
    
    correct = Y_pred == Y
    correct = np.sum(correct)
    accuracy = float(correct)/float(m)
    print( 'Accuracy is %s%%' % (accuracy *100))
        
    return Y_pred, accuracy
