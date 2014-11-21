import math
import scipy
import libowl
import sys
import random
import numpy
import time

from libowl import *
from scipy.io import *
from random import shuffle
from numpy import reshape
from numpy import *

class Layer(object):

    def __init__(length):
        self.length = length
        self.bias = owl.zeros((length, 1))

def make_vector(x, y):
    result = []
    for i in xrange(x):
        if (i == y):
            result.append(1)
        else:
            result.append(0)
    return result

def init_network(bias, weights, data, label, minibatch_size = 256, num_minibatches = 235):
    weights.append(randn([11, 11, 3, 96], 0.0, 0.1));
    weights.append(randn([5, 5, 96, 256], 0.0, 0.1));
    weights.append(randn([3, 3, 256, 384], 0.0, 0.1));
    weights.append(randn([3, 3, 384, 384], 0.0, 0.1));
    weights.append(randn([3, 3, 384, 256], 0.0, 0.1));
    weights.append(randn([4096, 9216], 0.0, 0.1));
    weights.append(randn([4096, 4096], 0.0, 0.1));
    weights.append(randn([1000, 4096], 0.0, 0.1));

    bias.append(randn([96], 0.0, 0.1));
    bias.append(randn([256], 0.0, 0.1));
    bias.append(randn([384], 0.0, 0.1));
    bias.append(randn([384], 0.0, 0.1));
    bias.append(randn([256], 0.0, 0.1));
    bias.append(randn([4096, 1], 0.0, 0.1));
    bias.append(randn([4096, 1], 0.0, 0.1));
    bias.append(randn([1000, 1], 0.0, 0.1));

def print_training_accuracy(o, t, minibatch_size):
    predict = o.max_index(0)
    ground_truth = t.max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)

def relu(act):
    re_acts = act.reshape(toscale([act.size(0), act.size(1), 1, 1]))
    act = activation_forward(re_acts, activation_algo.relu)
    return act.reshape(toscale([act.size(0), act.size(1)]))

def train_network(bias, weights, data, label,
                  num_epochs = 100, num_train_samples = 1000000, minibatch_size = 256,
                  num_minibatches = 3907, dropout_rate = 0.5, eps_w = 1, eps_b = 1):
    eps_w = eps_w / minibatch_size
    eps_b = eps_b / minibatch_size
    cpuDevice = create_cpu_device()
    gpuDevice = [None] * 2
    gpuDevice[0] = create_gpu_device(0)
    gpuDevice[1] = create_gpu_device(1)

    num_layers = 17
    last = time.time()
    for i in xrange(num_epochs):
        cur = time.time()
        print "Epoch #", i, ", time: %s" % (cur - last)
        for j in xrange(num_minibatches / 2):
          dW = [None] * 16
          dB = [None] * 16
          for k in xrange(2):
            set_device(gpuDevice[k])
            acts = [None] * num_layers
            sens = [None] * num_layers

            # FF
            acts[0] = randn([227, 227, 3, minibatch_size], 0.0, 0.1) # data
            target = randn([1000, minibatch_size], 0.0, 0.1)
            #print "data generated: %s" % time.clock()

            acts[1] = conv_forward(acts[0], weights[0], bias[0], conv_info(0, 0, 4, 4)) # conv1
            acts[2] = activation_forward(acts[1], activation_algo.relu) # relu1
            acts[3] = pooling_forward(acts[2], pooling_info(3, 3, 2, 2, pooling_algo.max)) # pool1

            acts[4] = conv_forward(acts[3], weights[1], bias[1], conv_info(2, 2, 1, 1)) # conv2
            acts[5] = activation_forward(acts[4], activation_algo.relu) # relu2
            acts[6] = pooling_forward(acts[5], pooling_info(3, 3, 2, 2, pooling_algo.max)) # pool2

            acts[7] = conv_forward(acts[6], weights[2], bias[2], conv_info(1, 1, 1, 1)) # conv3
            acts[8] = activation_forward(acts[7], activation_algo.relu) # relu3
            acts[9] = conv_forward(acts[8], weights[3], bias[3], conv_info(1, 1, 1, 1)) # conv4
            acts[10] = activation_forward(acts[9], activation_algo.relu) # relu4

            acts[11] = conv_forward(acts[10], weights[4], bias[4], conv_info(1, 1, 1, 1)) # conv5
            acts[12] = activation_forward(acts[11], activation_algo.relu) # relu5
            acts[13] = pooling_forward(acts[12], pooling_info(3, 3, 2, 2, pooling_algo.max)) # pool5

            re_acts13 = acts[13].reshape(toscale([acts[13].size(0) * acts[13].size(1) * acts[13].size(2), minibatch_size]))

            acts[14] = (weights[5] * re_acts13).norm_arithmetic(bias[5], arithmetic.add) # fc6
            acts[14] = relu(acts[14]) # relu6
            mask6 = randb([4096, minibatch_size], dropout_rate)
            acts[14] = mult(acts[14], mask6) # drop6

            acts[15] = (weights[6] * acts[14]).norm_arithmetic(bias[6], arithmetic.add) # fc7
            acts[15] = relu(acts[15]) # relu7
            mask7 = randb([4096, minibatch_size], dropout_rate)
            acts[15] = mult(acts[15], mask7) # drop7

            acts[16] = (weights[7] * acts[15]).norm_arithmetic(bias[7], arithmetic.add) # fc8
            acts[16] = softmax(acts[16]) # prob

            sens[16] = acts[16] - target

            # BP
            d_act15 = libowl.mult(acts[15], 1 - acts[15])
            sens[15] = weights[7].trans() * sens[16]
            sens[15] = libowl.mult(sens[15], d_act15) # fc8

            d_act14 = libowl.mult(acts[14], 1 - acts[14])
            sens[14] = weights[6].trans() * sens[15]
            sens[14] = libowl.mult(sens[14], d_act14) # fc7

            d_act13 = libowl.mult(re_acts13, 1 - re_acts13)
            sens[13] = weights[5].trans() * sens[14]
            sens[13] = libowl.mult(sens[13], d_act13)
            sens[13] = sens[13].reshape(toscale([acts[13].size(0), acts[13].size(1), acts[13].size(2), acts[13].size(3)])) # fc6

            sens[12] = pooling_backward(sens[13], acts[13], acts[12], pooling_info(3, 3, 2, 2, pooling_algo.max)) # pool5
            sens[11] = activation_backward(sens[12], acts[12], acts[11], activation_algo.relu) # relu5
            sens[10] = conv_backward_data(sens[11], weights[4], conv_info(1, 1, 1, 1)) # conv5

            sens[9] = activation_backward(sens[10], acts[10], acts[9], activation_algo.relu) # relu4
            sens[8] = conv_backward_data(sens[9], weights[3], conv_info(1, 1, 1, 1)) # conv4
            sens[7] = activation_backward(sens[8], acts[8], acts[7], activation_algo.relu) # relu3
            sens[6] = conv_backward_data(sens[7], weights[2], conv_info(1, 1, 1, 1)) # conv3

            sens[5] = pooling_backward(sens[6], acts[6], acts[5], pooling_info(3, 3, 2, 2, pooling_algo.max)) # pool2
            sens[4] = activation_backward(sens[5], acts[5], acts[4], activation_algo.relu) # relu2
            sens[3] = conv_backward_data(sens[4], weights[1], conv_info(2, 2, 1, 1)) # conv2

            sens[2] = pooling_backward(sens[3], acts[3], acts[2], pooling_info(3, 3, 2, 2, pooling_algo.max)) # pool1
            sens[1] = activation_backward(sens[2], acts[2], acts[1], activation_algo.relu) # relu1
            sens[0] = conv_backward_data(sens[1], weights[0], conv_info(0, 0, 4, 4)) # conv1

            dW[k * 8 + 7] = eps_w * sens[16] * acts[15].trans()
            dB[k * 8 + 7] = eps_b * sens[16].sum(1)

            dW[k * 8 + 6] = eps_w * sens[15] * acts[14].trans()
            dB[k * 8 + 6] = eps_b * sens[15].sum(1)

            dW[k * 8 + 5] = eps_w * sens[14] * re_acts13.trans()
            dB[k * 8 + 5] = eps_b * sens[14].sum(1)

            dW[k * 8 + 4] = eps_w * conv_backward_filter(sens[11], acts[10], conv_info(1, 1, 1, 1))
            dB[k * 8 + 4] = eps_b * conv_backward_bias(sens[11])

            dW[k * 8 + 3] = eps_w * conv_backward_filter(sens[9], acts[8], conv_info(1, 1, 1, 1))
            dB[k * 8 + 3] = eps_b * conv_backward_bias(sens[9])

            dW[k * 8 + 2] = eps_w * conv_backward_filter(sens[7], acts[6], conv_info(1, 1, 1, 1))
            dB[k * 8 + 2] = eps_b * conv_backward_bias(sens[7])

            dW[k * 8 + 1] = eps_w * conv_backward_filter(sens[4], acts[3], conv_info(2, 2, 1, 1))
            dB[k * 8 + 1] = eps_b * conv_backward_bias(sens[4])

            dW[k * 8 + 0] = eps_w * conv_backward_filter(sens[1], acts[0], conv_info(0, 0, 4, 4))
            dB[k * 8 + 0] = eps_b * conv_backward_bias(sens[1])

            if ((j % 1) == 0):
                acts[-1].eval()
                #print_training_accuracy(acts[-1], target, minibatch_size)
                #print "eval done: %s" % time.clock()
          for k in xrange(8):
            weights[k] -= dW[k]
            weights[k] -= dW[k + 8]
            bias[k] -= dB[k]
            bias[k] -= dB[k + 8]

if __name__ == '__main__':
    initialize(sys.argv)
    device = create_cpu_device()
    set_device(device)
    weights = []
    bias = []
    data = []
    label = []
    init_network(bias, weights, data, label)
    train_network(bias, weights, data, label)
