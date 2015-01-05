#!/usr/bin/env python
""" This module contains element-wise operations on ndarray
"""
import libowl as _owl

def mult(x, y):
    """ Element-wise multiplication

    Args:
      x (owl.NArray): first ndarray
      y (owl.NArray): second ndarray

    Returns:
      owl.NArray: result after element-wise multiplication
    """
    return _owl.mult(x, y)

def exp(x):
    """ Exponential function

    Args:
      x (owl.NArray): input
      
    Returns:
      owl.NArray: result ndarray
    """
    return _owl.exp(x)

def ln(x):
    """ Ln function

    Args:
      x (owl.NArray): input

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.ln(x)

def sigm(x):
    """ Sigmoid function: 1 / (1 + exp(-x))

    Args:
      x (owl.NArray): input

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.sigm(x)

def relu(x):
    """ REctified Linear Unit: y = x if x >= 0; y = 0 if x < 0;

    Args:
      x (owl.NArray): input

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.relu(x)

def tanh(x):
    """ Hyperbolic tangent function

    Args:
      x (owl.NArray): input

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.tanh(x)

def sigm_back(y):
    """ Derivative of sigmoid function: y * (1 - y)

    Args:
      y (owl.NArray): error from higher layer

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.sigm_back(y)

def relu_back(y, x):
    """ Derivative of RELU function
    
    Args:
      y (owl.NArray): error from higher layer
      x (owl.NArray): input of forward pass

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.relu_back(y, x, x)

def tanh_back(y):
    """ Derivative of tanh function: sech^2(y)

    Args:
      y (owl.NArray): error from higher layer

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.tanh_back(y, y, y)
