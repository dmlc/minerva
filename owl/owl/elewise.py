#!/usr/bin/env python
""" This module contains element-wise operations on ndarray
"""
import libowl as _owl

def mult(x, y):
    """ Element-wise multiplication

    :param owl.NArray x: first ndarray
    :param owl.NArray y: second ndarray
    :return: result after element-wise multiplication
    :rtype: owl.NArray
    """
    return _owl.mult(x, y)

def exp(x):
    """ Exponential function

    :param owl.NArray x: input
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.exp(x)

def ln(x):
    """ Ln function

    :param owl.NArray x: input
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.ln(x)

def sigm(x):
    """ Sigmoid function: 1 / (1 + exp(-x))

    :param owl.NArray x: input
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.sigm(x)

def relu(x):
    """ REctified Linear Unit: y = x if x >= 0; y = 0 if x < 0;

    :param owl.NArray x: input
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.relu(x)

def tanh(x):
    """ Hyperbolic tangent function

    :param owl.NArray x: input
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.tanh(x)

def sigm_back(y):
    """ Derivative of sigmoid function: y * (1 - y)

    :param owl.NArray y: error from higher layer
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.sigm_back(y)

def relu_back(y, x):
    """ Derivative of RELU function

    :param owl.NArray y: error from higher layer
    :param owl.NArray x: input of forward pass
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.relu_back(y, x, x)

def tanh_back(y):
    """ Derivative of tanh function: sech^2(y)

    :param owl.NArray y: error from higher layer
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.tanh_back(y, y, y)
