#!/usr/bin/env python
""" This module contains operations for convolution, pooling and softmax

Attributes:
  soft_op (enum): Same enum type as cudnn's cudnnSoftmaxMode_t. Either ``soft_op.instance``
    or ``soft_op.channel``.

  pool_op (enum): Same enum type as cudnn's cudnnPoolingMode_t. Either ``pool_op.max`` 
    or ``pool_op.avg``.
"""
import libowl as _owl

#act_op = _owl.activation_algo
soft_op = _owl.softmax_algo
pool_op = _owl.pooling_algo

def softmax(x, op = soft_op.instance):
    """ Perform softmax on the given ndarray.

    Note that this function is currently only for softmax accross instances. And the last
    dimension of ``x`` should represent instances. If ``x`` is of four dimension, directly
    call the c++ routine. Otherwise, augment the number of dimension to four.

    Args:
      x (owl.NArray): the ndarray to be softmaxed
      op (owl.conv.soft_op): what type of softmax to perform

    Returns:
      owl.NArray: the ndarray after being softmaxed and of the same shape
    """
    if len(x.shape) == 4:
        return _owl.softmax_forward(x, op)
    else:
        ori_shape = list(x.shape)
        soft_shape = x.shape[0:-1] + [1 for i in range(4 - len(ori_shape))] + [x.shape[-1]]
        return _owl.softmax_forward(x.reshape(soft_shape), op).reshape(ori_shape)


class Convolver:
    """ Wrapper class for convolution.

    Attributes:
      param (libowl.ConvInfo): convolution parameters

    """
    def __init__(self, pad_h, pad_w, stride_v, stride_h):
        """ Constructor for Convolver class
        
        Args:
          pad_h (int): padding height
          pad_w (int): padding width
          stride_v (int): vertical stride length
          stride_h (int): horizontal stride length

        """
        ci = _owl.ConvInfo()
        ci.pad_height = pad_h
        ci.pad_width = pad_w
        ci.stride_vertical = stride_v
        ci.stride_horizontal = stride_h
        self.param = ci

    def ff(self, x, w, b):
        """ Feed-forward convolution

        Args:
          x (owl.NArray): input of the convolution
          w (owl.NArray): filters
          b (owl.NArray): bias of the convolution

        Returns:
          owl.NArray: result ndarray after forward convolution
        """
        return _owl.conv_forward(x, w, b, self.param)

    def bp(self, y, w):
        """ Backward convolution

        Args:
          y (owl.NArray): error of the convolution usually passed by higher layers
          w (owl.NArray): filters

        Returns:
          owl.NArray: result ndarray after backward convolution
        """
        return _owl.conv_backward_data(y, w, self.param)

    def weight_grad(self, y, x):
        """ Compute the gradient of filters

        Args:
          y (owl.NArray): error (sensitivity) passed by higher layer
          x (owl.NArray): input (activation) of lower layer

        Returns:
          owl.NArray: the gradient of filters
        """
        return _owl.conv_backward_filter(y, x, self.param)

    def bias_grad(self, y):
        """ Compute the gradient of bias

        Args:
          y (owl.NArray): error (sensitivity) passed by higher layer

        Returns:
          owl.NArray: the gradient of bias
        """
        return _owl.conv_backward_bias(y)

class Pooler:
    """ Wrapper class for pooling operations

    Attributes:
      param (libowl.PoolingInfo): pooling parameters
    """
    def __init__(self, h, w, stride_v, stride_h, op):
        """ Constructor for Pooler class

        Args:
          h (int): pooling height
          w (int): pooling width
          stride_v (int): vertical stride length
          stride_h (int): horizontal stride length
          op (owl.conv.pool_op): pooling type
        """
        pi = _owl.PoolingInfo()
        pi.height = h
        pi.width = w
        pi.stride_vertical = stride_v
        pi.stride_horizontal = stride_h
        pi.algorithm = op
        self.param = pi

    def ff(self, x):
        """ Forward propagation for pooling

        Args:
          x (owl.NArray): input ndarray of pooling

        Returns:
          owl.NArray: output ndarray after forward pooling
        """
        return _owl.pooling_forward(x, self.param)

    def bp(self, y, ff_y, ff_x):
        """ Backward propagation for pooling

        Args:
          y (owl.NArray): error (sensitivity) from higher-layer
          ff_y (owl.NArray): value after forward pooling
          ff_x (owl.NArray): value before forward pooling

        Returns:
          owl.NArray: output after backward pooling
        """
        return _owl.pooling_backward(y, ff_y, ff_x, self.param)
