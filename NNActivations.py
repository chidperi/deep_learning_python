# File name: NNActivations
# Copyright 2017 Chidambaram Periakaruppan
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import numpy as np


def sigmoid(x):
    '''

    Args:
        x(ndarray)

    Returns:
        ndarray: Sigmoid activation and derivative of x.

    '''

    y = 1. / (1 + np.exp(-x))
    deriv = np.multiply(y, 1. - y)

    return y, deriv


def relu(x):
    '''

    Args:
        x(ndarray)

    Returns:
        ndarray: Relu activation and derivative of x.

    '''
    y = np.maximum(0, x)
    deriv = (x > 0) * 1

    return y, deriv


def leaky_relu(x, leak=0.1):
    '''

    Args:
        x(ndarray)
        leak(int)

    Returns:
        ndarray: Leaky relu activation and derivative of x.

    '''
    y = np.maximum(leak * x, x)
    deriv = (x < 0) * leak
    deriv = deriv + (x > 0) * 1

    return y, deriv


activation_functions = {'relu': relu, 'leaky_relu': leaky_relu, 'sigmoid': sigmoid}
