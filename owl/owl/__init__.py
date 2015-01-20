#!/usr/bin/env python
""" Owl module: Python binding for Minerva library

The module encapsulates or directly maps some functions of Minerva's API to python.
Only system APIs are defined here. For convolution APIs, please refer to conv.py.
For element-wise operations, please refer to elewise.py. For other APIs such as member 
functions of ``owl.NArray``, please refer to the API document.

Note that Minerva is an dataflow system with lazy evaluation to construct dataflow graph
to extract parallelism within codes. All the operations like +-*/ of ``owl.NArray`` are
all `lazy` such that they are NOT evaluated immediatedly. Only when you try to pull the
content of an NArray outside Minerva's control (e.g. call ``to_numpy()``) will those operations
be executed concretely.

We implement two interfaces for swapping values back and forth between numpy and Minerva.
They are ``from_numpy`` and ``to_numpy``. So you could still use any existing codes
on numpy such as IO and visualization.
"""
import numpy as np
import libowl as _owl

def initialize(argv):
    """ Initialize Minerva System.

    Note:
      Must be called before calling any owl's API

    Args:
      argv (list str): commandline arguments
    """
    _owl.initialize(argv)

def create_cpu_device():
    """ Create device for running on CPU cores

    Returns:
      int: A unique id for cpu device
    """
    return _owl.create_cpu_device()

def create_gpu_device(which):
    """ Create device for running on GPU card

    Args:
      which (int): which GPU card the code would be run on

    Returns:
      int: A unique id for the device on that GPU card
    """
    return _owl.create_gpu_device(which)

def set_device(dev):
    """ Switch to the given device for running computations
    
    When ``set_device(dev)`` is called, all the subsequent codes will be run on ``dev``
    till another ``set_device`` is called.

    Args:
      dev (int): the id of the device (usually returned by create_xxx_device)
    """
    _owl.set_device(dev)

def zeros(shape):
    """ Create ndarray of zero values

    Args:
      shape (list int): shape of the ndarray to create

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.zeros(shape)

def ones(shape):
    """ Create ndarray of one values

    Args:
      shape (list int): shape of the ndarray to create

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.ones(shape)

def randn(shape, mu, var):
    """ Create a random ndarray using normal distribution

    Args:
      shape (list int): shape of the ndarray to create
      mu (float): mu
      var (float): variance

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.randn(shape, mu, var)

def randb(shape, prob):
    """ Create a random ndarray using bernoulli distribution

    Args:
      shape (list int): shape of the ndarray to create
      prob (float): probability for the value to be one

    Returns:
      owl.NArray: result ndarray
    """
    return _owl.randb(shape, prob)

def from_numpy(nparr):
    """ Create an owl.NArray from numpy.ndarray

    Note:
      The content will be directly copied to Minerva's memory system. However, due to
      the different priority when treating dimensions between numpy and Minerva. The 
      result owl.NArray's dimension will be `reversed`.

        >>> import numpy as np
        >>> import owl
        >>> a = np.zeros([200, 300, 50])
        >>> b = owl.from_numpy(a)
        >>> print b.shape
        [50, 300, 200]

    Args:
      nparr (numpy.ndarray): numpy ndarray

    Returns:
      owl.NArray: Minerva's ndarray
    """
    return _owl.from_numpy(np.require(nparr, dtype=np.float32, requirements=['C']))

def concat(narrays, concat_dim): 
    """  Concatenate NArrays according to concat_dim
    
    Args:
        narrays (owl.NArray): inputs for concatenation
        concat_dim (int): the dimension to concate

    Returns:
        owl.NArray: result of concatenator
    """
    return _owl.concat(narrays, concat_dim)





