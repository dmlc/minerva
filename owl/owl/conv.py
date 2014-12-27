#!/usr/bin/env python
import libowl as _owl

act_op = _owl.activation_algo
soft_op = _owl.softmax_algo
pool_op = _owl.pooling_algo
'''
ConvInfo = _owl.ConvInfo
PoolingInfo = _owl.PoolingInfo

def conv_info(pad_h, pad_w, stride_v, stride_h):
    ci = ConvInfo()
    ci.pad_height = pad_h
    ci.pad_width = pad_w
    ci.stride_vertical = stride_v
    ci.stride_horizontal = stride_h
    return ci

def pooling_info(h, w, stride_v, stride_h, op):
    pi = PoolingInfo()
    pi.height = h
    pi.width = w
    pi.stride_vertical = stride_v
    pi.stride_horizontal = stride_h
    pi.algorithm = op
    return pi

conv_forward = _owl.conv_forward
conv_backward_data = _owl.conv_backward_data
conv_backward_filter = _owl.conv_backward_filter
conv_backward_bias = _owl.conv_backward_bias
pooling_forward = _owl.pooling_forward
pooling_backward = _owl.pooling_backward

activation_forward = _owl.activation_forward
activation_backward = _owl.activation_backward

softmax_forward = _owl.softmax_forward
softmax_backward = _owl.softmax_backward
'''
def softmax(x, op = soft_op.instance):
    if len(x.shape) == 4:
        return _owl.softmax_forward(x, op)
    else:
        ori_shape = list(x.shape)
        soft_shape = x.shape[0:-1] + [1 for i in range(4 - len(ori_shape))] + [x.shape[-1]]
        return _owl.softmax_forward(x.reshape(soft_shape), op).reshape(ori_shape)


class Convolver:
    def __init__(self, pad_h, pad_w, stride_v, stride_h):
        ci = _owl.ConvInfo()
        ci.pad_height = pad_h
        ci.pad_width = pad_w
        ci.stride_vertical = stride_v
        ci.stride_horizontal = stride_h
        self.param = ci
    def ff(self, x, w, b):
        return _owl.conv_forward(x, w, b, self.param)
    def bp(self, y, w):
        return _owl.conv_backward_data(y, w, self.param)
    def weight_grad(self, y, x):
        return _owl.conv_backward_filter(y, x, self.param)
    def bias_grad(self, y):
        return _owl.conv_backward_bias(y)

class Pooler:
    def __init__(self, h, w, stride_v, stride_h, op):
        pi = _owl.PoolingInfo()
        pi.height = h
        pi.width = w
        pi.stride_vertical = stride_v
        pi.stride_horizontal = stride_h
        pi.algorithm = op
        self.param = pi
    def ff(self, x):
        return _owl.pooling_forward(x, self.param)
    def bp(self, y, ff_y, ff_x):
        return _owl.pooling_backward(y, ff_y, ff_x, self.param)
