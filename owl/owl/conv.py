#!/usr/bin/env python
import libowl as _owl

act_op = _owl.activation_algo
soft_op = _owl.softmax_algo
pool_op = _owl.pooling_algo
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

activation_forward = _owl.activation_forward
activation_backward = _owl.activation_backward

softmax_forward = _owl.softmax_forward
softmax_backward = _owl.softmax_backward

pooling_forward = _owl.pooling_forward
pooling_backward = _owl.pooling_backward
