import math
import scipy
import libowl
import sys
import random
import numpy

from libowl import *
from scipy.io import *
from random import shuffle
from numpy import reshape

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
    weights.append(randn([5, 5, 1, 8], 0.0, 0.1));
    weights.append(randn([5, 5, 8, 16], 0.0, 0.1));
    weights.append(randn([10, 256], 0.0, 0.1));

    bias.append(randn([8], 0.0, 0.1));
    bias.append(randn([16], 0.0, 0.1));
    bias.append(randn([10, 1], 0.0, 0.1));

    print 'init network'
    loader = scipy.io.loadmat('mnist_all.mat')
    tmp_data = []
    tmp_label = []
    for i in xrange(10):
        key = 'train' + str(i)
        lvector = make_vector(10, i)
        for j in xrange(len(loader[key])):
            tmp_data.append(loader[key][j])
            tmp_label.append(lvector)
    idx = []
    for i in xrange(len(tmp_data)):
        idx.append(i)
    random.shuffle(idx)

    s = 0
    t = minibatch_size
    for i in xrange(num_minibatches):
        minibatch_data = [tmp_data[j] for j in idx[s:t]]
        minibatch_label = [tmp_label[j] for j in idx[s:t]]

        data.append(reshape(minibatch_data, (t - s) * 784).tolist())
        label.append(reshape(minibatch_label, (t - s) * 10).tolist())

        s = s + minibatch_size
        t = min(t + minibatch_size, len(tmp_data))

def print_training_accuracy(o, t, minibatch_size):
    predict = o.reshape(toscale([10, minibatch_size])).max_index(0)
    ground_truth = t.reshape(toscale([10, minibatch_size])).max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)

def train_network(bias, weights, data, label,
                  num_epochs = 100, num_train_samples = 60000, minibatch_size = 256,
                  num_minibatches = 235, eps_w = 0.01, eps_b = 0.01):
    eps_w = eps_w / minibatch_size
    eps_b = eps_b / minibatch_size
    cpuDevice = create_cpu_device()
    gpuDevice = create_gpu_device(0)
    num_layers = 9
    for i in xrange(num_epochs):
        print "Epoch #", i
        for j in xrange(num_minibatches):
            set_device(cpuDevice)
            acts = [None] * num_layers
            sens = [None] * num_layers

            acts[0] = make_narray([28, 28, 1, minibatch_size], data[j])
            target = make_narray([10, 1, 1, minibatch_size], label[j])

            set_device(gpuDevice)
            acts[1] = conv_forward(acts[0], weights[0], bias[0], conv_info(0, 0, 1, 1))
            acts[2] = activation_forward(acts[1], activation_algo.relu)
            
            acts[3] = pooling_forward(acts[2], pooling_info(2, 2, 2, 2, pooling_algo.max))
            acts[4] = conv_forward(acts[3], weights[1], bias[1], conv_info(2, 2, 1, 1))
            acts[5] = activation_forward(acts[4], activation_algo.relu)
            
            acts[6] = pooling_forward(acts[5], pooling_info(3, 3, 3, 3, pooling_algo.max))
            re_acts6 = acts[6].reshape(toscale([acts[6].size(0) * acts[6].size(1) * acts[6].size(2), minibatch_size]))

            acts[7] = (weights[2] * re_acts6).norm_arithmetic(bias[2], arithmetic.add)
            acts[8] = softmax_forward(acts[7].reshape(toscale([10, 1, 1, minibatch_size])), softmax_algo.instance)
            sens[8] = acts[8] - target

            sens[7] = sens[8].reshape(toscale([10, minibatch_size]))
            sens[6] = (weights[2].trans() * sens[7]).reshape(toscale([acts[6].size(0), acts[6].size(1), acts[6].size(2), acts[6].size(3)]))
            sens[5] = pooling_backward(sens[6], acts[6], acts[5], pooling_info(3, 3, 3, 3, pooling_algo.max))
            sens[4] = activation_backward(sens[5], acts[5], acts[4], activation_algo.relu)
            sens[3] = conv_backward_data(sens[4], weights[1], conv_info(2, 2, 1, 1))

            sens[2] = pooling_backward(sens[3], acts[3], acts[2], pooling_info(2, 2, 2, 2, pooling_algo.max))
            sens[1] = activation_backward(sens[2], acts[2], acts[1], activation_algo.relu)

            weights[2] -= eps_w * sens[7] * re_acts6.trans();
            bias[2] -= eps_b * sens[7].sum(1);

            weights[1] -= eps_w * conv_backward_filter(sens[4], acts[3], conv_info(2, 2, 1, 1))
            bias[1] -= eps_b * conv_backward_bias(sens[4])

            weights[0] -= eps_w * conv_backward_filter(sens[1], acts[0], conv_info(0, 0, 1, 1))
            bias[0] -= eps_b * conv_backward_bias(sens[1])

            if ((j % 20) == 0):
                print_training_accuracy(acts[-1], target, minibatch_size)

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
