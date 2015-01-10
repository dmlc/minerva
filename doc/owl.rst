Owl APIs
===========

owl module
---------------

.. py:class:: owl.NArray

  .. py:attribute:: __add__

    x.__add__(y)<==>x+y

  .. py:attribute:: __sub__

    x.__sub__(y)<==>x-y

  .. py:attribute:: __mul__

    x.__mul__(y)<==>x*y

    If any of x and y is ``float`` number, it is an element-wise multiplication. Otherwise, it is a matrix multiplcation which is only allowed for 2-dimensional NArray (matrix).

  .. py:attribute:: __div__

    x.__div__(y)<==>x/y

  .. py:attribute:: __iadd__

    x.__add__(y)<==>x+=y

  .. py:attribute:: __isub__

    x.__sub__(y)<==>x-=y

  .. py:attribute:: __imul__

    x.__mul__(y)<==>x*=y

  .. py:attribute:: __idiv__

    x.__div__(y)<==>x/=y

  .. py:attribute:: shape
    
    A list of int that describes the shape of this NArray
  
  .. py:method:: sum(axis)
    
    Sum up value along the given axis

    :param int axis: the axis along which to do summation
    :return: the result ndarray
    :rtype: owl.NArray

  .. py:method:: max(axis)
    
    Calculate max value along the given axis

    :param int axis: the axis along which to do max
    :return: the result ndarray
    :rtype: owl.NArray

  .. py:method:: argmax(axis)

    Calculate the index of the max value along the given axis

    :param int axis: the axis along which to do max
    :return: the result ndarray
    :rtype: owl.NArray

  .. py:method:: count_zero()

    Return the number of zero elements in the NArray
    Note::

      This function is a non-lazy function.

    :return: number of zero elements
    :rtype: int

  .. py:method:: trans()
    
    Return the transposed NArray
    Note::

      Only allow transpose on 2-dimension NArray (matrix)

    :return: transposed NArray
    :rtype: owl.NArray

  .. py:method:: reshape(shape)

    Return the NArray that is of different shape but same contents.
    Note::
    
      Only shape of the same volume is allowed.

    :param shape: the new shape of the NArray
    :type shape: list int
    :return: the NArray with new shape
    :rtype: owl.NArray

  .. py:method:: wait_for_eval()

    Put this NArray into evaluation, but do NOT wait for the finish of evaluation. The function will return immediately.

  .. py:method:: start_eval()

    Put this NArray into evaluation and block the execution until this NArray is concretely evaluated.

  .. py:method:: to_numpy()

    Convert this NArray to numpy::ndarray
    Note::

      This function is a non-lazy function.

    :return: numpy's ndarray with the same contents
    :rtype: numpy::ndarray


.. automodule:: owl
    :members:
    :undoc-members:
    :show-inheritance:


owl.conv module
---------------

.. automodule:: owl.conv
    :members:
    :undoc-members:
    :show-inheritance:

owl.elewise module
------------------

.. automodule:: owl.elewise
    :members:
    :undoc-members:
    :show-inheritance:

owl.net module
--------------

.. automodule:: owl.net
    :members:
    :undoc-members:
    :show-inheritance:

.. Subpackage
 -----------
 .. toctree::
 owl.caffe: !!comment out
