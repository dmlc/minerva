#!/usr/bin/env python
""" Owl module: Python binding for Minerva library

The module encapsulates or directly maps some functions of Minerva's API to python.
Only system APIs are defined here. For convolution APIs, please refer to conv.py.
For element-wise operations, please refer to elewise.py. For other APIs such as member
functions of ``owl.NArray``, please refer to the API document.

Note that Minerva is an dataflow system with lazy evaluation to construct dataflow graph
to extract parallelism within codes. All the operations like ``+-*/`` of ``owl.NArray`` are
all *lazy* such that they are NOT evaluated immediatedly. Only when you try to pull the
content of an NArray outside Minerva's control (e.g. call ``to_numpy()``) will those operations
be executed concretely.

We implement two interfaces for swapping values back and forth between numpy and Minerva.
They are ``from_numpy`` and ``to_numpy``. So you could still use any existing codes
on numpy such as IO and visualization.
"""
import numpy as np
import libowl as _owl

def initialize(argv):
    """ Initialize Minerva System

    **Must be called before calling any owl's API**

    :param argv: commandline arguments
    :type argv: list str
    """
    _owl.initialize(argv)

def finalize():
    """ Finalize Minerva System

    :return: None
    """
    _owl.finalize()

def create_cpu_device():
    """ Create device for running on CPU cores

    :return: A unique id for cpu device
    :rtype: int
    """
    return _owl.create_cpu_device()

def create_gpu_device(which):
    """ Create device for running on GPU card

    :param int which: which GPU card the code would be run on
    :return: A unique id for the device on that GPU card
    :rtype: int
    """
    return _owl.create_gpu_device(which)

def get_gpu_device_count():
    """ Get the number of compute-capable GPU devices

    :return: Number of compute-capable GPU devices
    "rtype: int
    """
    return _owl.get_gpu_device_count()

def set_device(dev):
    """ Switch to the given device for running computations

    When ``set_device(dev)`` is called, all the subsequent codes will be run on ``dev``
    till another ``set_device`` is called.

    :param int dev: the id of the device (usually returned by create_xxx_device)
    """
    _owl.set_device(dev)

def zeros(shape):
    """ Create ndarray of zero values

    :param shape: shape of the ndarray to create
    :type shape: list int
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.zeros(shape)

def ones(shape):
    """ Create ndarray of one values

    :param shape: shape of the ndarray to create
    :type shape: list int
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.ones(shape)

def randn(shape, mu, var):
    """ Create a random ndarray using normal distribution

    :param shape: shape of the ndarray to create
    :type shape: list int
    :param float mu: mu
    :param float var: variance
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.randn(shape, mu, var)

def randb(shape, prob):
    """ Create a random ndarray using bernoulli distribution

    :param shape: shape of the ndarray to create
    :type shape: list int
    :param float prob: probability for the value to be one
    :return: result ndarray
    :rtype: owl.NArray
    """
    return _owl.randb(shape, prob)

def from_numpy(nparr):
    """ Create an owl.NArray from numpy.ndarray

    The content will be directly copied to Minerva's memory system. However, due to
    the different priority when treating dimensions between numpy and Minerva. The
    result ``owl.NArray``'s dimension will be *reversed*.

      >>> import numpy as np
      >>> import owl
      >>> a = np.zeros([200, 300, 50])
      >>> b = owl.from_numpy(a)
      >>> print b.shape
      [50, 300, 200]

    :param numpy.ndarray nparr: numpy ndarray
    :return: Minerva's ndarray
    :rtype: owl.NArray
    """
    return _owl.from_numpy(np.require(nparr, dtype=np.float32, requirements=['C']))

def print_profiler_result():
    """ Print result from execution profiler

    :return: None
    """
    _owl.print_profiler_result()

def reset_profiler_result():
    """ Reset execution profiler

    :return: None
    """
    _owl.reset_profiler_result()
